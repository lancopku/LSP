//
//  Latent Structured Perceptron Toolkit v1.0
//
//  Copyright(C) Xu Sun <xusun@pku.edu.cn>
//

using System;
using System.Collections.Generic;
using System.Text;
using System.IO;
using System.Collections;

namespace Program
{
    class richEdge
    {
        public static double train(dataSet X = null, dataSet XX = null)
        {
            //load data
            if (X == null && XX == null)
            {
                Console.WriteLine("\nreading training & test data...");
                Global.swLog.WriteLine("\nreading training & test data...");
                X = new dataSet(Global.fFeatureTrain, Global.fGoldTrain);
                XX = new dataSet(Global.fFeatureTest, Global.fGoldTest);
                MainClass.dataSizeScale(X);
                Console.WriteLine("data sizes (train, test): {0} {1}", X.Count, XX.Count);
                Global.swLog.WriteLine("data sizes (train, test): {0} {1}", X.Count, XX.Count);
            }
          
            double score = 0;

            toolboxRich tb = new toolboxRich(X);
            score = MainClass.baseTrain(XX, tb);
            resSummarize.write();
            //save model
            if (Global.save == 1)
                tb.Model.save(Global.fModel);
            
            return score;
        }

        public static double test()
        {
            dataSet X = new dataSet(Global.fFeatureTrain, Global.fGoldTrain);
            dataSet XX = new dataSet(Global.fFeatureTest, Global.fGoldTest);
            Global.swLog.WriteLine("data size (test): {0}", XX.Count);
            //load model for testing
            toolboxRich tb = new toolboxRich(X, false);

            List<double> scoreList = tb.test(XX, 0);

            double score = scoreList[0];
            Global.scoreListList.Add(scoreList);
            resSummarize.write();
            return score;
        }
    }

    class toolboxRich: toolbox
    {
        public toolboxRich(dataSet X, bool train = true)
        {
            if (train)//for training
            {
                _X = X;
                _fGene = new featureGeneRich(X);
                _model = new model(X, _fGene);
                _inf = new inferRich(this);
                initOptimizer();
            }
            else//for test
            {
                _X = X;
                _model = new model(Global.fModel);
                _fGene = new featureGeneRich(X, _model);
                _inf = new inferRich(this);
            }
        }

    }

    class featureGeneRich: featureGenerator
    {
        public featureGeneRich()
        {
        }

        //for training
        public featureGeneRich(dataSet X)
        {
            _nFeatureTemp = X.NFeatureTemp;
            Global.swLog.WriteLine("feature templates: {0}", _nFeatureTemp);

            _nTag = X.NTag;
            _nState = X.NTag * Global.nHiddenStatePerTag;
            int nNodeFeatures = _nFeatureTemp * _nState;
            int nEdgeFeatures = _nFeatureTemp * _nState * _nState;
            _backoffEdge = nNodeFeatures;
            _nCompleteFeature = nNodeFeatures + nEdgeFeatures;
            Global.swLog.WriteLine("complete features: {0}", _nCompleteFeature);
        }

        //for test
        public featureGeneRich(dataSet X, model m)
        {
            _nState = m.NState;
            _nFeatureTemp = X.NFeatureTemp;
            Global.swLog.WriteLine("feature templates: {0}", _nFeatureTemp);

            _nTag = X.NTag;
            int nNodeFeatures = _nFeatureTemp * _nState;
            int nEdgeFeatures = _nFeatureTemp * _nState * _nState;
            _backoffEdge = nNodeFeatures;
            _nCompleteFeature = nNodeFeatures + nEdgeFeatures;
            Global.swLog.WriteLine("complete features: {0}", _nCompleteFeature);
        }

        override public int getEdgeFeatID(int id, int sPre, int s)
        {
            return _backoffEdge + id * _nState * _nState + s * _nState + sPre;
        }

    }

    class inferRich: inference
    {
        public inferRich(toolbox tb)
            : base(tb)
        {
        }

        override public void getLogYY(model m, dataSeq x, int i, ref dMatrix YY, ref List<double> Y, bool takeExp, bool mask)
        {
            YY.set(0);
            listTool.listSet(ref Y, 0);

            float[] w = m.W;
            List<featureTemp> fList = _fGene.getFeatureTemp(x, i);
            int nState = m.NState;
            foreach(featureTemp ft in fList)
            {
                for (int s = 0; s < nState; s++)
                {
                    int f =_fGene.getNodeFeatID(ft.id,s);
                    Y[s] += w[f] * ft.val;
                }
            }
            if (i > 0)
            {
                foreach (featureTemp im in fList)
                {
                    for (int s = 0; s < nState; s++)
                    {
                        for (int sPre = 0; sPre < nState; sPre++)
                        {
                            int f = _fGene.getEdgeFeatID(im.id, sPre, s);
                            YY[sPre, s] += w[f] * im.val;
                        }
                    }
                }
            }
            double maskValue = double.MinValue;
            if (takeExp)
            {
                listTool.listExp(ref Y);
                YY.eltExp();
                maskValue = 0;
            }
            if (mask)
            {
                dMatrix statesPerNodes = m.getStatesPerNode(x);
                for (int s = 0; s < Y.Count; s++)
                {
                    if (statesPerNodes[i,s] == 0)
                        Y[s] = maskValue;
                }
            }
        }

    }

}