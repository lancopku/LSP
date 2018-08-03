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
using System.IO.Compression;

namespace Program
{
    class datasetList:List<dataSet>
    {
        protected int _nTag;
        protected int _nFeature;

        public datasetList(string fileFeature, string fileTag)
        {
            StreamReader srfileFeature = new StreamReader(fileFeature);
            StreamReader srfileTags = new StreamReader(fileTag);

            string txt = srfileFeature.ReadToEnd();
            txt = txt.Replace("\r", "");
            string[] fAry = txt.Split(Global.triLineEndAry, StringSplitOptions.RemoveEmptyEntries);
            
            txt = srfileTags.ReadToEnd();
            txt = txt.Replace("\r", "");
            string[] tAry = txt.Split(Global.triLineEndAry, StringSplitOptions.RemoveEmptyEntries);

            if (fAry.Length != tAry.Length)
                throw new Exception("error");

            _nFeature = int.Parse(fAry[0]);
            _nTag = int.Parse(tAry[0]);
            
            for (int i = 1; i < fAry.Length; i++)
            {
                string fBlock = fAry[i];
                string tBlock = tAry[i];
                dataSet ds = new dataSet();
                string[] fbAry = fBlock.Split(Global.biLineEndAry, StringSplitOptions.RemoveEmptyEntries);
                string[] tbAry = tBlock.Split(Global.biLineEndAry, StringSplitOptions.RemoveEmptyEntries);

                for (int k = 0; k < fbAry.Length; k++)
                {
                    string fm = fbAry[k];
                    string tm = tbAry[k];
                    dataSeq seq = new dataSeq();
                    seq.read(fm, tm);
                    ds.Add(seq);
                }
                Add(ds);
            }
            srfileFeature.Close();
            srfileTags.Close();
        }
    }

    class dataSet : List<dataSeq>
    {
        protected int _nTag;
        protected int _nFeatureTemp;

        public dataSet()
        {
        }

        public dataSet(int nTag, int nFeatureTemp)
        {
            _nTag = nTag;
            _nFeatureTemp = nFeatureTemp;
        }

        public dataSet(string fileFeature, string fileTags)
        {
            load(fileFeature, fileTags);
        }

        public dataSet randomShuffle()
        {
            List<int> ri = randomTool<int>.getShuffledIndexList(this.Count);
            dataSet X = new dataSet(this.NTag, this.NFeatureTemp);
            foreach (int i in ri)
                X.Add(this[i]);
            return X;
        }

        virtual public int[,] EdgeFeature()
        {
            throw new Exception("error");
        }

        virtual public void load(string fileFeature, string fileTag)
        {
            StreamReader srfileData = new StreamReader(fileFeature, Encoding.GetEncoding("utf-8"));
            StreamReader srfileTag = new StreamReader(fileTag, Encoding.GetEncoding("utf-8"));

            string txt = srfileData.ReadToEnd();
            txt = txt.Replace("\r", "");
            string[] fAry = txt.Split(Global.biLineEndAry, StringSplitOptions.RemoveEmptyEntries);

            txt = srfileTag.ReadToEnd();
            txt = txt.Replace("\r", "");
            string[] tAry = txt.Split(Global.biLineEndAry, StringSplitOptions.RemoveEmptyEntries);

            if (fAry.Length != tAry.Length)
                throw new Exception("error");

            _nFeatureTemp = int.Parse(fAry[0]);
            _nTag = int.Parse(tAry[0]);
            for (int i = 1; i < fAry.Length; i++)
            {
                string features = fAry[i];
                string tags = tAry[i];
                dataSeq seq = new dataSeq();
                seq.read(features, tags);
                Add(seq);
            }
            srfileData.Close();
            srfileTag.Close();
        }

        public int NTag
        {
            get { return _nTag; }
            set { _nTag = value; }
        }

        public int NFeatureTemp
        {
            get { return _nFeatureTemp; }
            set { _nFeatureTemp = value; }
        }

        public void setDataInfo(dataSet X)
        {
            _nTag = X.NTag;
            _nFeatureTemp = X.NFeatureTemp;
        }

    }

    class dataSeqTest
    {
        public dataSeq _x;
        public List<int> _yOutput;

        public dataSeqTest(dataSeq x, List<int> yOutput)
        {
            _x = x;
            _yOutput = yOutput;
        }
    }

    class dataSeq
    {
        protected List<List<featureTemp>> featureTemps = new List<List<featureTemp>>();
        protected List<int> yGold = new List<int>();
        protected dMatrix goldStatesPerNode = new dMatrix(); //no inited when reading data, but during FB of training
        /*
        records the matrix hiddenStatesNodes, 
        e.g.:
        0 0
        0 0
        0 1
        0 1
        1 0
        1 0
        when the tag sentence is 2,1 and every tag have two hidden states
        */

        public dataSeq()
        {
        }

        public dataSeq(List<List<featureTemp>> feat, List<int> y)
        {
            featureTemps = new List<List<featureTemp>>(feat);
            for (int i = 0; i < feat.Count; i++)
                featureTemps[i] = new List<featureTemp>(feat[i]);
            yGold = new List<int>(y);
        }

        public dataSeq(dataSeq x, int n, int length)
        {
            int end = 0;
            if (n + length < x.Count)
                end = n + length;
            else
                end = x.Count;
            for (int i = n; i < end; i++)
            {
                featureTemps.Add(x.featureTemps[i]);
                yGold.Add(x.yGold[i]);
            }
        }

        virtual public List<List<int>> getNodeFeature(int n)
        {
            throw new Exception("error");
        }

        virtual public void read(string a, int nState, string b)
        {
            throw new Exception("error");
        }

        public void read(string a, string b)
        {
            //features
            string[] lineAry = a.Split(Global.lineEndAry, StringSplitOptions.RemoveEmptyEntries);
            foreach (string im in lineAry)
            {
                List<featureTemp> nodeList = new List<featureTemp>();
                string[] imAry = im.Split(Global.commaAry, StringSplitOptions.RemoveEmptyEntries);
                foreach (string imm in imAry)
                {
                    if (imm.Contains("/"))
                    {
                        string[] biAry = imm.Split(Global.slashAry, StringSplitOptions.RemoveEmptyEntries);
                        featureTemp ft = new featureTemp(int.Parse(biAry[0]), double.Parse(biAry[1]));
                        nodeList.Add(ft);
                    }
                    else
                    {
                        featureTemp ft = new featureTemp(int.Parse(imm), 1);
                        nodeList.Add(ft);
                    }
                }
                featureTemps.Add(nodeList);
            }
            //yGold
            lineAry = b.Split(Global.commaAry, StringSplitOptions.RemoveEmptyEntries);
            foreach (string im in lineAry)
            {
                yGold.Add(int.Parse(im));
            }
        }

        virtual public int Count
        {
            get { return featureTemps.Count; }
        }

        public List<List<featureTemp>> getFeatureTemp()
        {
            return featureTemps;
        }

        public List<featureTemp> getFeatureTemp(int node)
        {
            return featureTemps[node];
        }

        public int getTags(int node)
        {
            return yGold[node];
        }

        public List<int> getTags()
        {
            return yGold;
        }

        public void setTags(List<int> tags)
        {
            if (tags.Count != yGold.Count)
                throw new Exception("error");
            for (int i = 0; i < tags.Count; i++)
                yGold[i] = tags[i];
        }

        public dMatrix GoldStatesPerNode
        {
            get{return goldStatesPerNode;}
            set { goldStatesPerNode = value; }
        }
            
    }














}