#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>
#include <mpi.h>

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

#define MAXCOUNT 30
#define WIDTH 10000
#define HEIGHT 10000

void compute_fractal_part(unsigned char *image_part, int start_row, int end_row,
                         float left, float top, float xscale, float yscale)
{
    float zx, zy, cx, tempx, cy;
    int x, y, count;

    for (y = start_row; y < end_row; y++) {
        for (x = 0; x < WIDTH; x++) {
            cx = x * xscale + left;
            cy = y * yscale + top;
            zx = zy = 0;
            count = 0;
            
            while ((zx * zx + zy * zy < 4) && (count < MAXCOUNT)) {
                tempx = zx * zx - zy * zy + cx;
                zy = 2 * zx * zy + cy;
                zx = tempx;
                count++;
            }

            int idx = ((y - start_row) * WIDTH + x) * 3;
            if (count == MAXCOUNT) {
                image_part[idx] = 0;
                image_part[idx+1] = 0;
                image_part[idx+2] = 0;
            } else {
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

    float left = -1.75f;
    float top = -0.25f;
    float xside = 2.0f;
    float yside = 2.0f;

    float xscale = xside / WIDTH;
    float yscale = yside / HEIGHT;

    // Divide o trabalho entre os processos
    int rows_per_process = HEIGHT / size;
    int remainder_rows = HEIGHT % size;
    
    int start_row = rank * rows_per_process + (rank < remainder_rows ? rank : remainder_rows);
    int end_row = start_row + rows_per_process + (rank < remainder_rows ? 1 : 0);
    
    int local_rows = end_row - start_row;
    int local_size = local_rows * WIDTH * 3;

    // Aloca memória para a parte local da imagem
    unsigned char *local_image = (unsigned char*)malloc(local_size);
    if (!local_image) {
        fprintf(stderr, "Processo %d: Falha ao alocar memória para imagem local!\n", rank);
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    gettimeofday(&start_computation, NULL);
    compute_fractal_part(local_image, start_row, end_row, left, top, xscale, yscale);
    gettimeofday(&end_computation, NULL);

    computation_time = (end_computation.tv_sec - start_computation.tv_sec) +
                      (end_computation.tv_usec - start_computation.tv_usec) / 1000000.0;
    printf("Processo %d: Computou %d linhas em %.4f segundos\n", 
           rank, local_rows, computation_time);

    unsigned char *full_image = NULL;
    int *recvcounts = NULL;
    int *displs = NULL;

    if (rank == 0) {
        full_image = (unsigned char*)malloc(WIDTH * HEIGHT * 3);
        if (!full_image) {
            fprintf(stderr, "Processo 0: Falha ao alocar memória para imagem completa!\n");
            MPI_Abort(MPI_COMM_WORLD, 1);
        }

        recvcounts = (int*)malloc(size * sizeof(int));
        displs = (int*)malloc(size * sizeof(int));

        // Calcula recvcounts e displs para cada processo
        int offset = 0;
        for (int i = 0; i < size; i++) {
            int i_rows = rows_per_process + (i < remainder_rows ? 1 : 0);
            recvcounts[i] = i_rows * WIDTH * 3;
            displs[i] = offset;
            offset += recvcounts[i];
        }
    }

    // Usa gather para coletar as partes da imagem de todos os processos
    MPI_Gatherv(local_image, local_size, MPI_UNSIGNED_CHAR,
                full_image, recvcounts, displs, MPI_UNSIGNED_CHAR,
                0, MPI_COMM_WORLD);

    // Processo 0 salva a imagem
    if (rank == 0) {
        if (!stbi_write_png("mandelbrot_mpi.png", WIDTH, HEIGHT, 3, full_image, WIDTH * 3)) {
            fprintf(stderr, "Processo 0: Erro ao salvar imagem\n");
        } else {
            printf("Processo 0: Imagem salva com sucesso\n");
        }

        free(full_image);
        free(recvcounts);
        free(displs);
    }

    free(local_image);

    gettimeofday(&end_total, NULL);
    total_time = (end_total.tv_sec - start_total.tv_sec) +
                (end_total.tv_usec - start_total.tv_usec) / 1000000.0;
    printf("Processo %d: Tempo total: %.4f segundos\n", rank, total_time);

    MPI_Finalize();
    return 0;
}