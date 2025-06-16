#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>
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
            // Calcula, para cada pixel, o ponto correspondente no plano complexo
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

int main(int argc, char **argv)
{
    int rank, size;
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    struct timeval start_total, end_total;
    struct timeval start_computation, end_computation;
    double computation_time, total_time;

    gettimeofday(&start_total, NULL);

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
    
    int local_rows = end_row - start_row;
    
    // Aloca memória para a parte da imagem que este processo irá calcular
    unsigned char *local_image = (unsigned char*)malloc(local_rows * WIDTH * 3);
    if (!local_image) {
        printf("Processo %d: Falha ao alocar memória para imagem local!\n", rank);
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    gettimeofday(&start_computation, NULL);
    compute_fractal_part(local_image, start_row, end_row, left, top, xscale, yscale);
    gettimeofday(&end_computation, NULL);

    computation_time = (end_computation.tv_sec - start_computation.tv_sec) +
                       (end_computation.tv_usec - start_computation.tv_usec) / 1000000.0;
    printf("Processo %d: Tempo de computação do fractal: %.4f seconds\n", rank, computation_time);

    // Processo 0 irá coletar todas as partes e salvar a imagem
    if (rank == 0) {
        // Aloca memória para a imagem completa
        unsigned char *full_image = (unsigned char*)malloc(WIDTH * HEIGHT * 3);
        if (!full_image) {
            printf("Processo 0: Falha ao alocar memória para imagem completa!\n");
            MPI_Abort(MPI_COMM_WORLD, 1);
        }
        
        // Copia a parte do processo 0
        memcpy(full_image, local_image, local_rows * WIDTH * 3);
        
        // Recebe as partes dos outros processos
        for (int src = 1; src < size; src++) {
            // Calcula as linhas que este processo tratou
            int src_start_row = src * rows_per_process;
            int src_end_row = src_start_row + rows_per_process;
            
            if (src < remainder_rows) {
                src_start_row += src;
                src_end_row += src + 1;
            } else {
                src_start_row += remainder_rows;
                src_end_row += remainder_rows;
            }
            
            int src_rows = src_end_row - src_start_row;
            MPI_Recv(full_image + src_start_row * WIDTH * 3, 
                     src_rows * WIDTH * 3, MPI_UNSIGNED_CHAR, 
                     src, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        }
        
        // Salva a imagem completa
        if (!stbi_write_png("mandelbrot_mpi.png", WIDTH, HEIGHT, 3, full_image, WIDTH * 3)) {
            printf("Processo 0: Erro ao salvar imagem\n");
        } else {
            printf("Processo 0: Imagem salva no arquivo mandelbrot_mpi.png\n");
        }
        
        free(full_image);
    } else {
        // Envia a parte calculada para o processo 0
        MPI_Send(local_image, local_rows * WIDTH * 3, 
                MPI_UNSIGNED_CHAR, 0, 0, MPI_COMM_WORLD);
    }

    free(local_image);

    gettimeofday(&end_total, NULL);

    total_time = (end_total.tv_sec - start_total.tv_sec) +
                 (end_total.tv_usec - start_total.tv_usec) / 1000000.0;
    printf("Processo %d: Tempo total de execução: %.4f segundos\n", rank, total_time);

    MPI_Finalize();
    return 0;
}