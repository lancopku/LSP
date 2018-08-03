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
    //feature template
    struct featureTemp
    {
        public readonly int id;//feature id
        public readonly double val;//feature value

        public featureTemp(int a, double b)
        {
            id = a;
            val = b;
        }
    }

    class featureGenerator
    {
        protected int _nFeatureTemp;
        protected int _nCompleteFeature;
        protected int _backoffEdge;
        protected int _nTag;
        protected int _nState;

        public featureGenerator()
        {
        }

        //for train
        public featureGenerator(dataSet X)
        {
            _nFeatureTemp = X.NFeatureTemp;
            Global.swLog.WriteLine("feature templates: {0}", _nFeatureTemp);

            int nTag = X.NTag;
            _nTag = nTag;
            _nState = nTag * Global.nHiddenStatePerTag;
            int nNodeFeature = _nFeatureTemp * _nState;
            int nEdgeFeature = _nState * _nState;
            _backoffEdge = nNodeFeature;
            _nCompleteFeature = nNodeFeature + nEdgeFeature;
            Global.swLog.WriteLine("complete features: {0}", _nCompleteFeature);
        }

        //for test
        public featureGenerator(dataSet X, model m)
        {
            _nState = m.NState;
            _nFeatureTemp = X.NFeatureTemp;
            Global.swLog.WriteLine("feature templates: {0}", _nFeatureTemp);

            _nTag = X.NTag;
            int nNodeFeature = _nFeatureTemp * _nState;
            int nEdgeFeature = _nState * _nState;
            _backoffEdge = nNodeFeature;
            _nCompleteFeature = nNodeFeature + nEdgeFeature;
            Global.swLog.WriteLine("complete features: {0}", _nCompleteFeature);
        }

        public List<featureTemp> getFeatureTemp(dataSeq x, int node)
        {
            return x.getFeatureTemp(node);
        }

        public int getNodeFeatID(int id, int s)
        {
            return id * _nState + s;
        }

        virtual public int getEdgeFeatID(int sPre, int s)
        {
            return _backoffEdge + s * _nState + sPre;
        }

        virtual public int getEdgeFeatID(int id, int sPre, int s)
        {
            throw new Exception("error");
        }

        virtual public void getFeatures(dataSeq x, int node, ref List<List<int>> nodeFeature, ref int[,] edgeFeature)
        {
            throw new Exception("error");
        }

        public int BackoffEdge { get { return _backoffEdge; } }

        public int NState { get { return _nState; } }

        public int NCompleteFeature { get { return _nCompleteFeature; } }
    }
}