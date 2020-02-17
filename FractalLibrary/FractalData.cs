using BaseX;

namespace FractalLibrary
{
    public enum FractalType
    {
        none,
        mandelbrot,
        julia,
        newton
    }

    public class FractalData
    {
        public FractalType fractalType;

        //If the Julia set is used, customize C value, if Mandelbrot customize Z value
        public double2 customCorZValue;

        public FractalData() { }
        public FractalData( FractalType type, double2 customValue )
        {
            fractalType = type;
            customCorZValue = customValue;
        }
    }
}
