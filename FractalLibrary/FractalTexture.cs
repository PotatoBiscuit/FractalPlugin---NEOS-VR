using BaseX;
using FrooxEngine;

namespace FractalLibrary
{
    class FractalTexture : ProceduralTexture
    {
        public readonly Sync<double2> min;
        public readonly Sync<double2> max;
        public readonly Sync<double2> customConstant;

        protected override void ClearTextureData()
        {
            ;
        }
    }
}
