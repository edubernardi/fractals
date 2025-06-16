#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

// Define o número máximo de iterações para o cálculo de cada ponto
#define MAXCOUNT 30

// Define resolução da imagem a ser gerada
#define WIDTH 10000
#define HEIGHT 10000

void fractal(unsigned char *image, float left, float top, float xside, float yside)
{
    float xscale, yscale, zx, zy, cx, tempx, cy;
    int x, y, count;

    // Calcula a escala do plano complexo baseado na resolução da imagem e região do plano escolhida
    xscale = xside / WIDTH;
    yscale = yside / HEIGHT;

    // Itera sobre cada pixel da imagem
    for (y = 0; y < HEIGHT; y++) {
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
            int idx = (y * WIDTH + x) * 3;
            if (count == MAXCOUNT) {
                // Preto para pixels dentro do conjunto de Mandelbrot
                image[idx] = 0;
                image[idx+1] = 0;
                image[idx+2] = 0;
            } else {
                // Gradiente colorida para pixels fora do conjunto de Mandelbrot
                float t = (float)count / MAXCOUNT;
                image[idx] = (unsigned char)(9 * (1-t) * t * t * t * 255);
                image[idx+1] = (unsigned char)(15 * (1-t) * (1-t) * t * t * 255);
                image[idx+2] = (unsigned char)(8.5 * (1-t) * (1-t) * (1-t) * t * 255);
            }
        }
    }
}

int main()
{
    struct timeval start_total, end_total;
    struct timeval start_computation, end_computation;
    double computation_time, total_time;

    gettimeofday(&start_total, NULL);

    // Define a região do plano complexo a ser visualizada
    float left = -1.75f;
    float top = -0.25f;
    float xside = 2.0f;
    float yside = 2.0f;

    // Aloca memória para a imagem em formato RGB, com 3 bytes por pixel
    unsigned char *image = (unsigned char*)malloc(WIDTH * HEIGHT * 3);
    if (!image) {
        printf("Falha ao alocar memória para imagem!\n");
        return 1;
    }

    gettimeofday(&start_computation, NULL);
    fractal(image, left, top, xside, yside);
    gettimeofday(&end_computation, NULL);

    computation_time = (end_computation.tv_sec - start_computation.tv_sec) +
                       (end_computation.tv_usec - start_computation.tv_usec) / 1000000.0;
    printf("Tempo de computação do fractal: %.4f segundos\n", computation_time);

    // Função para salvar a imagem como PNG
    if (!stbi_write_png("mandelbrot_seq.png", WIDTH, HEIGHT, 3, image, WIDTH * 3)) {
        printf("Erro ao salvar imagem\n");
    } else {
        printf("Imagem salva no arquivo mandelbrot.png\n");
    }

    free(image);

    gettimeofday(&end_total, NULL);

    total_time = (end_total.tv_sec - start_total.tv_sec) +
                 (end_total.tv_usec - start_total.tv_usec) / 1000000.0;
    printf("Tempo total de execução (inclui salvar a imagem): %.4f segundos\n", total_time);

    return 0;
}