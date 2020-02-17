using FrooxEngine.LogiX;
using FrooxEngine;

namespace FractalLibrary
{
    [Category("LogiX/Fractals")]
    class NewtonInput : LogixNode
    {
        public readonly Output<FractalData> fractalData;

        protected override void OnStart()
        {
            base.OnStart();
            fractalData.Value = new FractalData();
            ChangeFractalData();
        }

        protected override void OnInputChange()
        {
            base.OnInputChange();
            ChangeFractalData();
        }

        private void ChangeFractalData()
        {
            fractalData.Value.fractalType = FractalType.newton;
        }
    }
}
