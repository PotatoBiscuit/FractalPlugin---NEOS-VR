﻿using CodeX;
using FrooxEngine;
using BaseX;
using System.Runtime.InteropServices;
using System;

namespace FractalLibrary
{
    [Category("Assets/Procedural Textures")]
    class MandelbrotTexture : FractalTexture
    {
        private int height;
        private int width;

        private double rangeX;
        private double rangeY;
        private float maxIterations = 256f;

        public static class GpuRef
        {
            private const string DllFilePath = @".\ComputeFractalGpu.dll";

            [DllImport(DllFilePath, CallingConvention = CallingConvention.Cdecl)]
            private extern static void computeMandelbrot(int[] iterArray, int width, int height, double minX, double minY, double rangeX, double rangeY);

            public static void ComputeMandelbrot(int[] iterArray, int width, int height, double minX, double minY, double rangeX, double rangeY)
            {
                computeMandelbrot(iterArray, width, height, minX, minY, rangeX, rangeY);
            }
        }

        protected override void OnAwake()
        {
            //Set defaults in case these inputs don't exist
            min.Value = new double2(-2f, -1f);
            max.Value = new double2(1f, 1f);

            height = 1000;
            width = 1500;

            Size.Value = new int2(width, height);
            Mipmaps.Value = false;
            Format.Value = TextureFormat.ARGB32;
        }

        protected override void UpdateTextureData(Bitmap2D tex2D)
        {
            height = tex2D.Size.y;
            width = tex2D.Size.x;
            rangeX = max.Value.x - min.Value.x;
            rangeY = max.Value.y - min.Value.y;

            try
            {
                CreateMandelbrotGpu(tex2D);
            }
            catch (DllNotFoundException)
            {
                CreateMandelbrot(tex2D);
            }
        }

        private void CreateMandelbrotGpu(Bitmap2D tex2D)
        {
            int[] iterArray = new int[width * height];
            GpuRef.ComputeMandelbrot(iterArray, width, height, min.Value.x, min.Value.y, rangeX, rangeY);
            float colorToUse;
            for (int y = 0; y < height; y++)
            {
                for (int x = 0; x < width; x++)
                {
                    colorToUse = iterArray[y * width + x] / maxIterations;
                    tex2D.SetPixel(x, y, new color(colorToUse, colorToUse, colorToUse)); //depending on the number of iterations, color a pixel.
                }
            }
        }

        private void CreateMandelbrot(Bitmap2D tex2D)
        {
            double2 coords;
            int iterations;
            double realZ = 0;
            double imagZ = 0;
            double realZ2 = 0;
            double imagZ2 = 0;
            float colorToUse;

            for (int y = 0; y < height; y++)
            {
                for (int x = 0; x < width; x++)
                {
                    coords = ConvertPixToCoords(x, y);
                    realZ = coords[0];
                    imagZ = coords[1];

                    iterations = 0;
                    while (iterations < maxIterations)
                    {
                        iterations++;
                        realZ2 = realZ * realZ;
                        imagZ2 = imagZ * imagZ;
                        if (realZ2 + imagZ2 > 4)
                        {
                            break;
                        }
                        imagZ = 2 * realZ * imagZ + coords[1];
                        realZ = realZ2 - imagZ2 + coords[0];
                    }
                    colorToUse = iterations / maxIterations;
                    tex2D.SetPixel(x, y, new color(1 - colorToUse, 1 - colorToUse, 1 - colorToUse)); //depending on the number of iterations, color a pixel.
                }
            }
        }

        private double2 ConvertPixToCoords(int x, int y)
        {
            return new double2(min.Value.x + (rangeX / width) * x, min.Value.y + (rangeY / height) * y);
        }

        protected override void ClearTextureData()
        {
            ;
        }
    }
}
