using CodeX;
using FrooxEngine;
using BaseX;

namespace FractalLibrary
{
    [Category("Assets/Procedural Textures")]
    class JuliaTexture : FractalTexture
    {
        private int height;
        private int width;

        private double rangeX;
        private double rangeY;
        private float maxIterations = 256f;

        protected override void OnAwake()
        {
            //Set defaults in case these inputs don't exist
            min.Value = new double2(-2f, -1f);
            max.Value = new double2(1f, 1f);
            customConstant.Value = new double2(0, 0);

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

            CreateJulia(tex2D);
        }

        private void CreateJulia(Bitmap2D tex2D)
        {
            double2 coords;
            int iterations;
            double realZ = 0;
            double imagZ = 0;
            double realZ2 = 0;
            double imagZ2 = 0;
            double realC = customConstant.Value[0];
            double imagC = customConstant.Value[1];
            float colorToUse;

            for (int y = 0; y < height; y++)
            {
                for (int x = 0; x < width; x++)
                {
                    coords = ConvertPixToCoords(x, y);
                    realZ = coords[0] + realC;
                    imagZ = coords[1] + imagC;

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
                        imagZ = 2 * realZ * imagZ + imagC;
                        realZ = realZ2 - imagZ2 + realC;
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
