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
using System.Threading;

namespace Program
{
    class Global
    {
        //default values
        public static string runMode = "train";//train (normal training), train.rich (training with rich edge features), test, cv (cross validation), cv.rich
        public static string modelOptimizer = "lsp-2.paf";//sp.avg/paf/naive, lsp-n.avg/paf/naive (the method introduced in [Sun+ TKDE 2013])
        public static int random = 1;//0 for 0-initialization of model weights, 1 for random init of model weights
        public static string evalMetric = "tok.acc";//tok.acc (token accuracy), str.acc (string accuracy), f1 (F1-score)
        public static double trainSizeScale = 1;//for scaling the size of training data
        public static int ttlIter = 50;//# of training iterations
        public static string outFolder = "out";
        public static int save = 1;//save model file
        public static bool rawResWrite = true;
        public static int nCV = 4;//automatic #-fold cross validation
        public static List<List<dataSeqTest>> threadXX;
        public static int nThread = 10;
        public static bool tuneInit = true;

        //general
        public const double tuneSplit = 0.8;//size of data split for tuning
        public static bool debug = false;//some debug code will run in debug mode
        //perceptron
        public static int nPaf = 2;
        public static int sig = 1;
        //tuning
        public const int nTuneRound = 5;
        public const int iterTuneWeightInit = 20;//default 20

        //global variables
        public static bool tuneWeightInit = false;
        public static int nHiddenStatePerTag = 1;
        public static baseHashMap<int, string> chunkTagMap = new baseHashMap<int, string>();
        public static string metric;
        public static float[] tmpW = null;
        public static float[] optimW = null;
        public static double ttlScore = 0;
        public static string outDir = "";
        public static List<List<double>> scoreListList = new List<List<double>>();
        public static List<double> timeList = new List<double>();
        public static List<double> errList = new List<double>();
        public static List<double> diffList = new List<double>();
        public static int glbIter = 0;
        public static double diff = 1e100;//relative difference from the previous object value, for convergence test
        public static int countWithIter = 0;
        public static StreamWriter swTune;
        public static StreamWriter swLog;
        public static StreamWriter swResRaw;
        public static StreamWriter swOutput;
        public const string fTune = "tune.txt";
        public const string fLog = "trainLog.txt";
        public const string fResSum = "summarizeResult.txt";
        public const string fResRaw = "rawResult.txt";
        public const string fFeatureTrain = "ftrain.txt";
        public const string fGoldTrain = "gtrain.txt";
        public const string fFeatureTest = "ftest.txt";
        public const string fGoldTest = "gtest.txt";
        public const string fOutput = "outputTag.txt";
        public const string fModel = "model/model.txt";
        public const string modelDir = "model/";
        public static char[] lineEndAry = { '\n' };
        public static string[] biLineEndAry = { "\n\n" };
        public static string[] triLineEndAry = { "\n\n\n" };
        public static char[] barAry = { '-' };
        public static char[] dotAry = { '.'};
        public static char[] underlnAry = { '_' };
        public static char[] commaAry = { ',' };
        public static char[] tabAry = { '\t' };
        public static char[] vertiBarAry = { '|' };
        public static char[] colonAry = { ':' };
        public static char[] blankAry = { ' ' };
        public static char[] starAry = { '*' };
        public static char[] slashAry = { '/' };
 
        public static void reinitGlobal()
        {
            diff = 1e100;
            countWithIter = 0;
            glbIter = 0;
            Global.nPaf = 2;
            Global.sig = 1;
        }

        public static void globalCheck()
        {
            //get # of hidden states
            string[] tmpAry = Global.modelOptimizer.Split(dotAry, StringSplitOptions.RemoveEmptyEntries);
            if (tmpAry.Length != 2)
                throw new Exception("error");
            string[] tmpAry2 = tmpAry[0].Split(barAry, StringSplitOptions.RemoveEmptyEntries);
            string model = tmpAry2[0];
            if (model == "sp")
                Global.nHiddenStatePerTag = 1;
            else
            {
                if (tmpAry2.Length != 2)
                    throw new Exception("error");
                Global.nHiddenStatePerTag = int.Parse(tmpAry2[1]);
            }

            if (runMode.Contains("test"))
                ttlIter = 1;

            if (evalMetric == "f1")
                getChunkTagMap();

            if (evalMetric == "f1")
                metric = "f-score";
            else if (evalMetric == "tok.acc")
                metric = "token-accuracy";
            else if (evalMetric == "str.acc")
                metric = "string-accuracy";
            else throw new Exception("error");

            if (Global.trainSizeScale <= 0)
                throw new Exception("error");
            if (Global.ttlIter <= 0)
                throw new Exception("error");
        }

        public static void printGlobals()
        {
            swLog.WriteLine("mode: {0}", Global.runMode);
            swLog.WriteLine("modelOptimizer: {0}", Global.modelOptimizer);
            swLog.WriteLine("random: {0}", Global.random);
            swLog.WriteLine("nHiddenPerTag: {0}", Global.nHiddenStatePerTag);
            swLog.WriteLine("evalMetric: {0}", Global.evalMetric);
            swLog.WriteLine("trainSizeScale: {0}", Global.trainSizeScale);
            swLog.WriteLine("ttlIter: {0}", Global.ttlIter);
            swLog.WriteLine("outFolder: {0}", Global.outFolder);
            swLog.Flush();
        }

        //the system must know the B (begin-chunk), I (in-chunk), O (out-chunk) information for computing f-score
        //since such BIO information is task-dependent, tagIndex.txt is required
        static void getChunkTagMap()
        {
            chunkTagMap.Clear();

            //read the labelMap.txt for chunk tag information
            StreamReader sr = new StreamReader("tagIndex.txt");
            string a = sr.ReadToEnd();
            a = a.Replace("\r", "");
            string[] ary = a.Split(Global.lineEndAry, StringSplitOptions.RemoveEmptyEntries);
            foreach (string im in ary)
            {
                string[] imAry = im.Split(Global.blankAry, StringSplitOptions.RemoveEmptyEntries);
                int index = int.Parse(imAry[1]);
                string[] tagAry = imAry[0].Split(Global.starAry, StringSplitOptions.RemoveEmptyEntries);
                string tag = tagAry[tagAry.Length - 1];//the last tag is the current tag
                //merge I-tag/O-tag: no need to use diversified I-tag/O-tag in computing F-score
                if (tag.StartsWith("I"))
                    tag = "I";
                if (tag.StartsWith("O"))
                    tag = "O";
                chunkTagMap[index] = tag;
            }

            sr.Close();
        }

    }

}
