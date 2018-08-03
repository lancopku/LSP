//
//  Latent Structured Perceptron Toolkit v1.0
//
//  Copyright(C) Xu Sun <xusun@pku.edu.cn>
//

using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.IO;
using System.Diagnostics;
using System.Threading;
using System.Threading.Tasks;

namespace Program
{
    class optimPerc : optimizer
    {
        float[] _sumW, _tmpW;
        bool recoverFlag;

        public optimPerc(toolbox tb)
        {
            _model = tb.Model;
            _X = tb.X;
            _inf = tb.Inf;
            _fGene = tb.FGene;

            _sumW = new float[_model.W.Length];
            _tmpW = new float[_model.W.Length];
            recoverFlag = false;

            //reinit globals
            Global.reinitGlobal();
        }

        public override double optimize()
        {
            structPercTrain();
            Global.swLog.Flush();
            return 0;
        }

        //for efficiency in dealing with sparse features: accumulated weights A_n = n*w0 + n*f1 + (n-1)*f2 + (n-2)*f3 +...
        void structPercTrain()
        {
            if (recoverFlag)
            {
                _model.W = _tmpW;
                recoverFlag = false;
            }

            int wUpdate = 0;
            int xsize = _X.Count;

            List<int> ri;//debug
            if (Global.glbIter <= 1)
            {
                ri = randomTool<int>.getSortedIndexList(xsize);
                Console.WriteLine("sorted index!");
            }
            else
                ri = randomTool<int>.getShuffledIndexList(xsize);

            //re-init temp vector in every iteration
            for (int i = 0; i < _tmpW.Length; i++)
                _tmpW[i] = xsize * _model.W[i];

            for (int k = 0; k < xsize; k++)
            {
                int j = ri[k];
                dataSeq x = _X[j];

                List<dMatrix> YYlist = new List<dMatrix>(), maskYYlist = new List<dMatrix>();
                List<List<double>> Ylist = new List<List<double>>(), maskYlist = new List<List<double>>();
                _inf.getYYandY(_model, x, YYlist, Ylist, maskYYlist, maskYlist);

                //find the output Viterbi hidden path
                List<int> outStates = new List<int>();
                _inf.decodeViterbi_train(_model, x, YYlist, Ylist, outStates);

                //find the oracle Viterbi hidden path
                List<int> goldStates = new List<int>();
                _inf.decodeViterbi_train(_model, x, maskYYlist, maskYlist, goldStates);

                //update the weights		
                if (compare(outStates, goldStates) == false)
                {
                    if (Global.runMode.Contains("rich"))
                        updateWeights_richEdge(x, outStates, goldStates, _model.W, _tmpW, xsize, k);
                    else
                        updateWeights(x, outStates, goldStates, _model.W, _tmpW, xsize, k);
                    wUpdate++;
                }

                Global.countWithIter++;
            }

            //accumulate for the averaged weight vector
            for (int i = 0; i < _sumW.Length; i++)
            {
                _sumW[i] += _tmpW[i];
            }
            Global.swLog.WriteLine("iter{0}    #update={1}", Global.glbIter, wUpdate);

            if (Global.modelOptimizer.EndsWith("paf"))//the method introduced in [Sun+ TKDE 2013]
            {
                //PAF iteration
                if (Global.glbIter == Global.nPaf)
                {
                    Global.swLog.WriteLine("re-init weigths");
                    Global.swLog.WriteLine("sig {0}", Global.sig);
                    Global.swLog.WriteLine("nPaf {0}", Global.nPaf);
                    Global.swLog.WriteLine("glbIter {0}", Global.glbIter);

                    Global.sig++;
                    Global.nPaf += Global.sig;

                    //averaging
                    for (int i = 0; i < _sumW.Length; i++)
                    {
                        _model.W[i] = _sumW[i] / (float)Global.countWithIter;
                    }
                }
                //normal iteration
                else
                {
                    //backup model weights
                    for (int i = 0; i < _sumW.Length; i++)
                        _tmpW[i] = _model.W[i];
                    //averaging (for test)
                    for (int i = 0; i < _sumW.Length; i++)
                        _model.W[i] = _sumW[i] / (float)Global.countWithIter;
                    //a flag for recover
                    recoverFlag = true;
                }
            }
            else if (Global.modelOptimizer.EndsWith("avg"))
            {
                //backup model weights
                for (int i = 0; i < _sumW.Length; i++)
                    _tmpW[i] = _model.W[i];
                //averaging (for test)
                for (int i = 0; i < _sumW.Length; i++)
                    _model.W[i] = _sumW[i] / (float)Global.countWithIter;
                //a flag for recover
                recoverFlag = true;
            }
        }

        bool compare(List<int> a, List<int> b)
        {
            for (int i = 0; i < a.Count; i++)
                if (a[i] != b[i])
                {
                    return false;
                }

            return true;
        }

        void updateWeights(dataSeq x, List<int> outStates, List<int> goldStates, float[] w, float[] accumW, int xsize, int k)
        {
            for (int n = 0; n < x.Count; n++)
            {
                int outState = outStates[n];
                int goldState = goldStates[n];

                //update the weights and accumulative weights
                foreach(featureTemp im in _fGene.getFeatureTemp(x, n))
                {
                    int f = _fGene.getNodeFeatID(im.id, outState);
                    float fv = (float)im.val;
                    w[f] -= fv;
                    float t = xsize - k;
                    accumW[f] -= t * fv;

                    f = _fGene.getNodeFeatID(im.id, goldState);
                    w[f] += fv;
                    accumW[f] += t * fv;
                }

                if (n > 0)
                {
                    int f = _fGene.getEdgeFeatID(outStates[n - 1], outState);
                    float fv = 1;
                    w[f] -= fv;
                    float t = xsize - k;
                    accumW[f] -= t * fv;

                    f = _fGene.getEdgeFeatID(goldStates[n - 1], goldState);
                    w[f] += fv;
                    accumW[f] += t * fv;
                }
            }
        }

        void updateWeights_richEdge(dataSeq x, List<int> outStates, List<int> goldStates, float[] w, float[] accumW, int nSamples, int k)
        {
            for (int n = 0; n < x.Count; n++)
            {
                int outState = outStates[n];
                int goldState = goldStates[n];

                List<featureTemp> featureTemps = _fGene.getFeatureTemp(x, n);
                //update the weights and accumulative weights
                foreach(featureTemp im in featureTemps)
                {
                    int f = _fGene.getNodeFeatID(im.id, outState);
                    float fv = (float)im.val;
                    w[f] -= fv;
                    float t = nSamples - k;
                    accumW[f] -= t * fv;

                    f = _fGene.getNodeFeatID(im.id, goldState);
                    w[f] += fv;
                    accumW[f] += t * fv;
                }

                if (n > 0)
                {
                    foreach (featureTemp im in featureTemps)
                    {
                        int f = _fGene.getEdgeFeatID(im.id, outStates[n - 1], outState);
                        float fv = 1;
                        w[f] -= fv;
                        float t = nSamples - k;
                        accumW[f] -= t * fv;

                        f = _fGene.getEdgeFeatID(im.id, goldStates[n - 1], goldState);
                        w[f] += fv;
                        accumW[f] += t * fv;
                    }
                }
            }
        }  
    }
}
