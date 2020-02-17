using System;
using System.Drawing;
using System.Numerics;
using BaseX;

namespace ImageViewer
{
    public class FractalViewer
    {
        public int maxIterations = 400; // increasing this will give you a more detailed fractal
        public int height = 1000;
        public int width = 1500;
        public double minX = -1;
        public double minY = -1;
        public double rangeX = 4;
        public double rangeY = 4;

        /*
        public Bitmap MBrot(string funcToExec)
        {
            Type thisType = GetType();
            Bitmap texture = new Bitmap(width, height);

            double2 coords;
            int iterations;
            double realZ = 0;
            double imagZ = 0;
            double realZ2 = 0;
            double imagZ2 = 0;
            for(int y = 0; y < height; y++)
            {
                for(int x = 0; x < width; x++)
                {
                    coords = convertPixToCoords(x, y);
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
                    texture.SetPixel(x, y, Color.FromArgb(iterations - 1 % maxIterations, 0, 0, 0)); //depending on the number of iterations, color a pixel.
                }
            }
            
            return texture;
        }
        */

        public Bitmap CreateNewton()
        {
            Complex coords;
            int iterations;
            Complex epsilon;
            int colorToUse;
            double magSquared;

            Bitmap texture = new Bitmap(width, height);

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
                    
                    if ( iterations < maxIterations)
                    {
                        colorToUse = 0;
                        if (Math.Abs(coords.Real + .5) < .001 && Math.Abs(coords.Imaginary + .866) < .001)
                        {
                            colorToUse = 0;
                        }
                        else if (Math.Abs(coords.Real - 1) < .001 && Math.Abs(coords.Imaginary) < .001)
                        {
                            colorToUse = 100;
                        }
                        else if (Math.Abs(coords.Real + .5) < .001 && Math.Abs(coords.Imaginary - .866) < .001)
                        {
                            colorToUse = 255;
                        }
                        texture.SetPixel(x, y, Color.FromArgb(colorToUse, colorToUse, colorToUse)); //depending on the number of iterations, color a pixel.
                    }
                }
            }
            return texture;
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
            return new Complex(minX + (rangeX / width) * x, minY + (rangeY / height) * y);
        }
    }
}
