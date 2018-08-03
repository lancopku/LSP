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
    class belief
    {
        public List<List<double>> belState;
        public List<dMatrix> belEdge;
        public double Z;

        public belief(int nNodes, int nStates)
        {
            List<double>[] dAry = new List<double>[nNodes];
            belState = new List<List<double>>(dAry);
            for (int i = 0; i < nNodes; i++)
            {
                double[] dAry2 = new double[nStates];
                belState[i] = new List<double>(dAry2);
            }

            dMatrix[] dAry3 = new dMatrix[nNodes];
            belEdge = new List<dMatrix>(dAry3);
            for (int i = 1; i < nNodes; i++)
                belEdge[i] = new dMatrix(nStates, nStates);

            belEdge[0] = null;
            Z = 0;
        }
    }

    class inference
    {
        protected optimizer _optim;
        protected featureGenerator _fGene;

        public inference(toolbox tb)
        {
            _optim = tb.Optim;
            _fGene = tb.FGene;
        }

        virtual public void getLogYY(model m, dataSeq x, int i, ref dMatrix YY, ref List<double> Y, bool takeExp, bool mask)
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
                    int f = _fGene.getNodeFeatID(ft.id, s);
                    Y[s] += w[f] * ft.val;
                }
            }
            if (i > 0)
            {
                for (int s = 0; s < nState; s++)
                {
                    for (int sPre = 0; sPre < nState; sPre++)
                    {
                        int f = _fGene.getEdgeFeatID(sPre, s);
                        YY[sPre, s] += w[f];
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
                    if (statesPerNodes[i, s] == 0)
                        Y[s] = maskValue;
                }
            }
        }

        public void getYYandY(model m, dataSeq x, List<dMatrix> YYlist, List<List<double>> Ylist, List<dMatrix> maskYYlist, List<List<double>> maskYlist)
        {
            int nNodes = x.Count;
            //int nTag = m.NTag;
            int nState = m.NState;
            double[] dAry = new double[nState];
            bool mask = false;

            try
            {
                //Global.rwlock.AcquireReaderLock(Global.readWaitTime);

                for (int i = 0; i < nNodes; i++)
                {
                    dMatrix YYi = new dMatrix(nState, nState);
                    List<double> Yi = new List<double>(dAry);
                    //compute the Mi matrix
                    getLogYY(m, x, i, ref YYi, ref Yi, false, mask);
                    YYlist.Add(YYi);
                    Ylist.Add(Yi);

                    maskYYlist.Add(new dMatrix(YYi));
                    maskYlist.Add(new List<double>(Yi));
                }

                //Global.rwlock.ReleaseReaderLock();
            }
            catch (ApplicationException)
            {
                Console.WriteLine("read out time!");
            }

            //get the masked YY and Y
            double maskValue = double.MinValue;
            dMatrix statesPerNodes = m.getStatesPerNode(x);
            for (int i = 0; i < nNodes; i++)
            {
                List<double> Y = maskYlist[i];
                List<int> tagList = x.getTags();
                for (int s = 0; s < Y.Count; s++)
                {
                    if (statesPerNodes[i, s] == 0)
                        Y[s] = maskValue;
                }
            }
        }

        //fast viterbi decode without probability
        public void decodeViterbi_train(model m, dataSeq x, List<dMatrix> YYlist, List<List<double>> Ylist, List<int> tags)
        {
            int nNode = x.Count;
            int nState = m.NState;
            Viterbi viter = new Viterbi(nNode, nState);

            for (int i = 0; i < nNode; i++)
            {
                viter.setScores(i, Ylist[i], YYlist[i]);
            }

            double numer = viter.runViterbi(ref tags, false);
        }

        //fast viterbi decode without probability
        public void decodeViterbi_test(model m, dataSeq x, List<int> tags)
        {
            tags.Clear();

            int nNode = x.Count;
            int nState = m.NState;
            dMatrix YY = new dMatrix(nState, nState);
            double[] dAry = new double[nState];
            List<double> Y = new List<double>(dAry);
            Viterbi viter = new Viterbi(nNode, nState);

            for (int i = 0; i < nNode; i++)
            {
                getLogYY(m, x, i, ref YY, ref Y, false, false);
                viter.setScores(i, Y, YY);
            }

            List<int> states = new List<int>();
            double numer = viter.runViterbi(ref states, false);
            for (int i = 0; i < states.Count; i++)
            {
                int tag = m.hStateToTag(states[i]);
                tags.Add(tag);
            }
        }

    }
}