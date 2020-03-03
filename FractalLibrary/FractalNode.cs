using FrooxEngine.LogiX;
using FrooxEngine;
using BaseX;

namespace FractalLibrary
{
    [Category("LogiX/Fractals")]
    class CreateComplexFractal : LogixNode
    {
        public readonly Input<FractalData> fractalData;

        public readonly Input<int> height;
        public readonly Input<int> width;

        public readonly Input<double2> min;
        public readonly Input<double2> max;

        private Slot visual;
        private FractalTexture resultTexture;
        private UnlitMaterial resultMat;
        private QuadMesh graph;
        private MeshRenderer graphRenderer;

        [ImpulseTarget]
        public void CreateComplexFractalTexture()
        {
            if( fractalData.IsConnected )
            {
                if ( visual == null )
                {
                    CreateNewVisual();
                }
                else
                {
                    UpdateVisual();
                }
            }
        }

        private void UpdateVisual()
        {
            FractalType fractalType = fractalData.Evaluate(new FractalData(FractalType.none, new double2(0, 0))).fractalType;
            if ((resultTexture is MandelbrotTexture && fractalType != FractalType.mandelbrot) ||
                (resultTexture is JuliaTexture && fractalType != FractalType.julia) ||
                (resultTexture is NewtonTexture && fractalType != FractalType.newton))
            {
                visual.Destroy();
                CreateNewVisual();
            }
            else
            {
                ChangeTextureAttributes();
                AdjustGraphAttributes();
            }
        }

        private void CreateNewVisual()
        {
            visual = Slot.AddSlot("ComplexFractalVisual");

            FractalType fractalType = fractalData.Evaluate(new FractalData(FractalType.none, new double2(0, 0))).fractalType;
            if ( fractalType == FractalType.mandelbrot )
            {
                resultTexture = visual.AttachComponent<MandelbrotTexture>();
            }
            else if ( fractalType == FractalType.julia )
            {
                resultTexture = visual.AttachComponent<JuliaTexture>();
            }
            else if ( fractalType == FractalType.newton )
            {
                resultTexture = visual.AttachComponent<NewtonTexture>();
            }
            else
            {
                return;
            }
            ChangeTextureAttributes();

            resultMat = visual.AttachComponent<UnlitMaterial>(false, null);
            resultMat.BlendMode.Value = BlendMode.Transparent;
            resultMat.Texture.Target = resultTexture;
            graph = visual.AttachQuad(new float2(0.6f, 0.4f), resultMat, true);
            AdjustGraphAttributes();

            graphRenderer = visual.AttachComponent<MeshRenderer>();
            graphRenderer.Materials.Add();
            graphRenderer.Materials[0] = resultMat;
            graphRenderer.Mesh.Target = graph;
        }
        
        private void ChangeTextureAttributes()
        {
            resultTexture.Size.Value = new int2(width.Evaluate(1500), height.Evaluate(1000));
            resultTexture.min.Value = min.Evaluate(new double2(-2f, -1f));
            resultTexture.max.Value = max.Evaluate(new double2(1f, 1f));

            double2 customValue = fractalData.Evaluate(new FractalData(FractalType.none, new double2(0, 0))).customCorZValue;
            resultTexture.customConstant.Value = customValue;
        }
        
        private void AdjustGraphAttributes()
        {
            visual.LocalPosition = new float3(0.0f, 0.175f + (.175f * height.Evaluate(1000) / 1000), 0.0f);
            graph.Size.Value = new float2(width.Evaluate(1500)/1000f*.3f, height.Evaluate(1000)/1000f*.3f);
        }
    }
}
