#include <stdio.h>
#include <stdlib.h>
#include <omp.h>

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

// Define o número máximo de iterações para o cálculo de cada ponto
#define MAXCOUNT 30

// Define resolução da imagem a ser gerada
#define WIDTH 10000
#define HEIGHT 10000

void fractal(unsigned char *image, float left, float top, float xside, float yside)
{
    float xscale, yscale;
    int x, y;

    // Calcula a escala do plano complexo baseado na resolução da imagem e região do plano escolhida
    xscale = xside / WIDTH;
    yscale = yside / HEIGHT;

    // Paraleliza o loop externo, sobre o eixo y (vertical).
    // A configuração 'schedule(dynamic)' é recomendada porque as áreas podem ter diferentes complexidades de cálculo,
    // como as bordas do conjunto de Mandelbrot, que exigem mais iterações.
    // O eixo x (horizontal) é privado para cada thread.
    #pragma omp parallel for private(x) schedule(dynamic)
    for (y = 0; y < HEIGHT; y++) {
        for (x = 0; x < WIDTH; x++) {
            float zx = 0.0f, zy = 0.0f;
            float cx = x * xscale + left;
            float cy = y * yscale + top;
            int count = 0;
            
            // Itera até que o ponto escape ou atinja o número máximo de iterações
            while ((zx * zx + zy * zy < 4.0f) && (count < MAXCOUNT)) {
                float tempx = zx * zx - zy * zy + cx;
                zy = 2.0f * zx * zy + cy;
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
    double start_total_time = omp_get_wtime();

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

    // Usa o máximo de threads disponíveis no sistema, se não for explicitamente definido
    omp_set_num_threads(omp_get_max_threads());
    
    printf("Gerando imagem com %d threads...\n", omp_get_max_threads());
    
    double start_computation_time = omp_get_wtime();
    
    fractal(image, left, top, xside, yside);

    double end_computation_time = omp_get_wtime();
    printf("Tempo de computação do fractal: %.4f segundos\n", end_computation_time - start_computation_time);

    // Função para salvar a imagem como PNG
    if (!stbi_write_png("mandelbrot_omp.png", WIDTH, HEIGHT, 3, image, WIDTH * 3)) {
        printf("Erro ao salvar imagem\n");
    } else {
        printf("Imagem salva no arquivo mandelbrot_omp.png\n");
    }

    free(image);

    double end_total_time = omp_get_wtime();
    printf("Tempo total de execução (inclui salvar a imagem): %.4f segundos\n", end_total_time - start_total_time);

    return 0;
}