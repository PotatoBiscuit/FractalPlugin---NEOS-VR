using CodeX;
using FrooxEngine;
using BaseX;
using System.Numerics;
using System;

namespace FractalLibrary
{
    [Category("Assets/Procedural Textures")]
    class NewtonTexture : FractalTexture
    {
        private int height;
        private int width;

        private double rangeX;
        private double rangeY;
        private float maxIterations = 400f;

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
            tex2D.Clear(color.White);
            height = tex2D.Size.y;
            width = tex2D.Size.x;
            rangeX = max.Value.x - min.Value.x;
            rangeY = max.Value.y - min.Value.y;

            CreateNewton(tex2D);
        }

        private void CreateNewton(Bitmap2D tex2D)
        {
            Complex coords;
            int iterations;
            Complex epsilon;
            float colorToUse;
            double magSquared;

            const double cutoff = 0.00000000001;

            for (int y = 0; y < height; y++)
            {
                for (int x = 0; x < width; x++)
                {
                    coords = ConvertPixToCoords(x, y);
                    iterations = 0;

                    do
                    {
                        epsilon = -(F(coords) / dFdx(coords));
                        coords += epsilon;
                        iterations++;
                        magSquared = Math.Pow(Complex.Abs(epsilon), 2);
                    }
                    while (magSquared > cutoff && iterations < maxIterations);

                    if( iterations < maxIterations)
                    {
                        colorToUse = 0;
                        if (Math.Abs(coords.Real + .5) < .001 && Math.Abs(coords.Imaginary + .866) < .001)
                        {
                            colorToUse = 0;
                        }
                        else if (Math.Abs(coords.Real - 1) < .001 && Math.Abs(coords.Imaginary) < .001)
                        {
                            colorToUse = .5f;
                        }
                        else if (Math.Abs(coords.Real + .5) < .001 && Math.Abs(coords.Imaginary - .866) < .001)
                        {
                            colorToUse = 1;
                        }
                        tex2D.SetPixel(x, y, new color(colorToUse, colorToUse, colorToUse)); //depending on the number of iterations, color a pixel.
                    }
                }
            }
        }

        private Complex F(Complex Z)
        {
            return Complex.Pow(Z, 3) - 1;
        }

        private Complex dFdx(Complex Z)
        {
            return 3 * Complex.Pow(Z, 2);
        }

        private Complex ConvertPixToCoords(int x, int y)
        {
            return new Complex(min.Value.x + (rangeX / width) * x, min.Value.y + (rangeY / height) * y);
        }

        protected override void ClearTextureData()
        {
            ;
        }
    }
}
