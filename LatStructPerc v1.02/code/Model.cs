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
    class model
    {
        int _nTag;
        int _nHiddenState;
        float[] _w;

        public model(string file)
        {
            if(File.Exists(file))
                load(file);
        }

        public model(dataSet X, featureGenerator fGen)
        {
            _nTag = X.NTag;
            _nHiddenState = Global.nHiddenStatePerTag * _nTag;
            //default value is 0
            if (Global.random == 0)
            {
                _w = new float[fGen.NCompleteFeature];
            }
            else if (Global.random == 1)
            {
                List<float> randList = randomDoubleTool.getRandomList_float(fGen.NCompleteFeature);
                _w = randList.ToArray();
                if (Global.tuneWeightInit)
                {
                    if (Global.tmpW == null)
                        Global.tmpW = new float[_w.Length];
                    _w.CopyTo(Global.tmpW, 0);
                }
            }
            else if (Global.random == 2)
            {
                _w = new float[fGen.NCompleteFeature];
                Global.optimW.CopyTo(_w, 0);
            }
            else throw new Exception("error");
        }

        public model(model m, bool wCopy)
        {
            _nTag = m.NTag;
            _nHiddenState = m.NState;

            _w = new float[m.W.Length];
            if (wCopy)
            {
                m.W.CopyTo(_w, 0);
            }
        }

        public void load(string file)
        {
            StreamReader sr = new StreamReader(file);
            string txt = sr.ReadToEnd();
            txt = txt.Replace("\r", "");
            string[] ary = txt.Split(Global.lineEndAry, StringSplitOptions.RemoveEmptyEntries);
            _nHiddenState = int.Parse(ary[0]);
            _nTag = int.Parse(ary[1]);
            int wsize = int.Parse(ary[2]);
            _w = new float[wsize];
            for (int i = 3; i < ary.Length; i++)
            {
                _w[i - 3] = float.Parse(ary[i]);
            }
            if (_w.Length != wsize)
                throw new Exception("error");

            sr.Close();
        }

        public void save(string file)
        {
            StreamWriter sw = new StreamWriter(file);
            sw.WriteLine(_nHiddenState);
            sw.WriteLine(_nTag);
            sw.WriteLine(_w.Length);
            foreach (double im in _w)
            {
                sw.WriteLine(im.ToString("f4"));
            }
            sw.Close();
        }

        public float[] W
        {
            get { return _w; }
            set
            {
                if (_w == null)
                {
                    float[] ary = new float[value.Length];
                }
                value.CopyTo(_w, 0);
            }             
        }

        public dMatrix getStatesPerNode(dataSeq x)
        {
            int n = x.Count;
            dMatrix spn = x.GoldStatesPerNode;
            if (spn == null || spn.R == 0)
            {
                List<int> tList = x.getTags();
                spn = new dMatrix(tList.Count, _nHiddenState);
                for (int i = 0; i < tList.Count; i++)
                {
                    int tag = tList[i];
                    List<int> states = tagToHiddenStates(tag);
                    foreach (int s in states)
                        spn[i, s] = 1;
                }
                x.GoldStatesPerNode = spn;
            }
            return spn;
        }

        public List<int> tagToHiddenStates(int tag)
        {
            List<int> states = new List<int>();
            int nStatePerTag = _nHiddenState / _nTag;
            int start = tag * nStatePerTag;
            for (int s = start; s < start + nStatePerTag; s++)
                states.Add(s);

            return states;
        }

        public int hStateToTag(int s)
        {
            int nStatePerTag = _nHiddenState / _nTag;
            return s / nStatePerTag;
        }

        public int NState 
        { 
            get { return _nHiddenState; }
            set
            {
                _nHiddenState = value;
            }
        }

        public int NTag
        {
            get { return _nTag; }
            set
            {
                _nTag = value;
            }
        }

    }
}
