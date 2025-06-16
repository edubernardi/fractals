#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

// Define o número máximo de iterações para o cálculo de cada ponto
#define MAXCOUNT 30

// Define resolução da imagem a ser gerada
#define WIDTH 10000
#define HEIGHT 10000

void compute_fractal_part(unsigned char *image_part, int start_row, int end_row, 
                         float left, float top, float xscale, float yscale)
{
    float zx, zy, cx, tempx, cy;
    int x, y, count;

    for (y = start_row; y < end_row; y++) {
        for (x = 0; x < WIDTH; x++) {
            // Calcula o ponto correspondente no plano complexo
            cx = x * xscale + left;
            cy = y * yscale + top;
            zx = zy = 0;
            count = 0;
            
            // Itera até que o ponto escape ou atinja o número máximo de iterações
            while ((zx * zx + zy * zy < 4) && (count < MAXCOUNT)) {
                tempx = zx * zx - zy * zy + cx;
                zy = 2 * zx * zy + cy;
                zx = tempx;
                count++;
            }

            // Define a cor do pixel baseado no número de iterações
            int idx = ((y - start_row) * WIDTH + x) * 3;
            if (count == MAXCOUNT) {
                // Preto para pixels dentro do conjunto de Mandelbrot
                image_part[idx] = 0;
                image_part[idx+1] = 0;
                image_part[idx+2] = 0;
            } else {
                // Gradiente colorida para pixels fora do conjunto de Mandelbrot
                float t = (float)count / MAXCOUNT;
                image_part[idx] = (unsigned char)(9 * (1-t) * t * t * t * 255);
                image_part[idx+1] = (unsigned char)(15 * (1-t) * (1-t) * t * t * 255);
                image_part[idx+2] = (unsigned char)(8.5 * (1-t) * (1-t) * (1-t) * t * 255);
            }
        }
    }
}

int main(int argc, char *argv[])
{
    int rank, size;
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    // Define a região do plano complexo a ser visualizada
    float left = -1.75f;
    float top = -0.25f;
    float xside = 2.0f;
    float yside = 2.0f;

    // Calcula a escala do plano complexo
    float xscale = xside / WIDTH;
    float yscale = yside / HEIGHT;

    // Divide o trabalho entre os processos
    int rows_per_process = HEIGHT / size;
    int remainder_rows = HEIGHT % size;
    
    // Determina as linhas que cada processo irá computar
    int start_row = rank * rows_per_process;
    int end_row = start_row + rows_per_process;
    
    // Distribui as linhas restantes para os primeiros processos
    if (rank < remainder_rows) {
        start_row += rank;
        end_row += rank + 1;
    } else {
        start_row += remainder_rows;
        end_row += remainder_rows;
    }
    
    // Cada processo aloca memória apenas para sua parte da imagem
    int part_height = end_row - start_row;
    unsigned char *image_part = (unsigned char*)malloc(part_height * WIDTH * 3);
    if (!image_part) {
        printf("Process %d: Memory allocation failed!\n", rank);
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    // Cada processo computa sua parte do fractal
    compute_fractal_part(image_part, start_row, end_row, left, top, xscale, yscale);

    // Processo 0 irá coletar todas as partes e salvar a imagem
    if (rank == 0) {
        unsigned char *full_image = (unsigned char*)malloc(WIDTH * HEIGHT * 3);
        if (!full_image) {
            printf("Process 0: Memory allocation for full image failed!\n");
            MPI_Abort(MPI_COMM_WORLD, 1);
        }

        // Copia a parte do processo 0
        memcpy(full_image, image_part, part_height * WIDTH * 3);
        
        // Recebe as partes dos outros processos
        for (int src = 1; src < size; src++) {
            // Determina as linhas que o processo src computou
            int src_start_row = src * rows_per_process;
            int src_end_row = src_start_row + rows_per_process;
            if (src < remainder_rows) {
                src_start_row += src;
                src_end_row += src + 1;
            } else {
                src_start_row += remainder_rows;
                src_end_row += remainder_rows;
            }
            int src_part_height = src_end_row - src_start_row;
            
            MPI_Recv(full_image + src_start_row * WIDTH * 3, 
                     src_part_height * WIDTH * 3, 
                     MPI_UNSIGNED_CHAR, 
                     src, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        }

        // Salva a imagem completa
        if (!stbi_write_png("mandelbrot_mpi.png", WIDTH, HEIGHT, 3, full_image, WIDTH * 3)) {
            printf("Erro ao salvar imagem\n");
        } else {
            printf("Imagem salva no arquivo mandelbrot_mpi.png\n");
        }

        free(full_image);
    } else {
        // Outros processos enviam suas partes para o processo 0
        MPI_Send(image_part, part_height * WIDTH * 3, 
                MPI_UNSIGNED_CHAR, 0, 0, MPI_COMM_WORLD);
    }

    free(image_part);
    MPI_Finalize();
    return 0;
}