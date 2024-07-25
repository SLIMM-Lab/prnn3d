
log =
{
  pattern = "*.info";
  file    = "-$(CASE_NAME).log";
};

control = 
{
  fgMode  = false; //true;
runWhile = "i < 1001";
};

userInput =
{
  modules = [ "input" ];

  input = 
  {
    type = "ANNInput";
    file = "tr_propgp_080.data"; 
    input = [0, 1, 2, 3, 4, 5, 6, 7, 8];
    output = [9, 10, 11, 12, 13, 14, 15, 16, 17];
    dts = [18];
    outNormalizer = "bounds";
  };
};

model =
{
  type = "Neural";
  rseed = 771;
  layers = ["input", "lowert", "lower", "mat","output"]; 
  weightSharing = [0, 0, 1, 0, 0];
  input = {
	  type = "Dense";
          size = 9;
	  //debug = true;
	};
  lowert = {
            type = "Block";
size = 72;
	    init = "glorot";
	    activateWeights = true;
	    activationWeights = "softplusdiag";
            //debug = true;
	    };
  lower = {
          type = "Block";
size = 72;
	  init = "glorot";
         // lower = true;
	  activateWeights = true;
	  activationWeights = "softplusdiag";
	  //debug = true;
	  };
   mat = 
   {
	    type = "MatRec";
            matList = ["egp", "bonet"];	 
size = 72;
size = 72;
nModels = [6, 2];
	    init = "glorot";
	    //useBias = true;
	    //debug = true;
    egp =
    {
        type   = "EGP";

        state  = "3D"; // 3D, PLANE_STRAIN or AXISYMMETRIC
        nn = true;
        dim    = 3; // if axisymm then dim = 2

        strHardening = "NEOHOOKEAN"; // EDWARDSVILGIS or NEOHOOKEAN

        // peek material
        Gr      = 14.2000000e0;
        t0a     = 1.38600000e0; // 1.386
        t0b     = 1.38600000e0;  
        SSa     = 3.000000e0; 
        SSb     = 0.000000e0; 
        r0a     = 0.95000e0; 
        r1a     = 1.000000e0; 
        r2a     = -5.000000e0; 
        r0b     = 0.95000e0; 
        r1b     = 1.000000e0; 
        r2b     = -5.000000e0;  
        ma      = 0.080000e0; 
        mb      = 0.080000e0; 
        k       = 2600.e0; // 2600 
        mode    = 1; 
        nom     = 1; // num of alpha modes
        nam     = 1; // total num of modes

        // G     = [ 721.05  ,  275.88  ,   31.768 ,   60.192 ,   49.951 ,   43.472 ,
        //           31.35  ,   29.26  ,   34.903 ,   57.893 ,   53.295 ,   41.8   ,
        //           39.083 ,    3.1977,   36.575 ,    2.3617];

        // h0    = [  7.59000000e+21,   4.25100000e+20,   1.28520000e+19,
        //            9.21600000e+18,   2.95620000e+18,   9.96000000e+17,
        //            2.76000000e+17,   9.93600000e+16,   4.59900000e+16,
        //            1.13580000e+16,   4.37820000e+14,   1.43700000e+13,
        //            5.63580000e+11,   1.92540000e+10,   9.19800000e+08,
        //            2.48040000e+07];

        G       = [ 1045.5225]; //,   400.026,    46.0636,    87.2784]; //,    72.429 //       63.0344,    45.4575,    42.427 ,    50.6093,    83.9449,            //       77.2777,    60.61  ,    56.6703,     4.6367,    53.0338,                //        3.4245];

        h0    = [  7.590e+21]; //,   8.502e+16,   2.570e+14,   1.843e+13]; //,   5.912e+12,  //  1.992e+12,   5.520e+11,   1.987e+11,   9.198e+10,   2.272e+10,     //   8.756e+8,   2.874e+07,   1.127e+06,   3.851e+04,   1.840e+03,                 //   4.961e+01];
      };
     
    bonet =
    {
        type   = "OrthotropicBonet"; // OrthotropicBonet
        dim    = 3;
        nn     = true;
        Ea     = 125.e3; // modulus in fiber direction 125.e3
        E      = 15.e3;  // modulus in transverse direction
        Ga     = 45.e3;
        nua    = .05;
        nu     = .3;
        fibdir = [0.,0.,1.];
      };
 };
  output = {
          type = "Sparse";
          size = 9;
          activateWeights = true;
          activationWeights = "abs";
          pruning = true;
          symmetric = true;
          //debug = true;
         // useBias = false;
  };
};

userModules =
{

  modules = [ "solver", "pred"]; //, "graphtr", "graphval" ];

  solver =
  {
    type = "Adam";
    loss = "squarederror";
    miniBatch = 2;
    selComp = [0, 1, 2, 4, 5, 8];
    precision = 1.e-4;
    nRestarts = 1;
    maxIter = 1;
    alpha = 0.001;
    skipFirst = 100;
subset = 72;
rseed = 771;
  };

  pred =
  {
    type = "ANNOutput";
    filename = "$(CASE_NAME)";
    writeEvery = 50;
    format = "lines";
    selComp = [0, 1, 2, 4, 5, 8];
    print = "weights | inputs | outputs";
    minValues = [2.7762e-2, 841e-2, 0.0]; //0.0, 0.0, -1.9824e-4, 4.372e-4];
    maxValues = [8.3069e-2, 8.3069e-2, 6.5034e-2];//, 8.3069e-2, 6.5034e-2, 8.3069e-2 ];
    steps = [4.372e-4, 4.372e-4,-1.9824e-4];//, 4.372e-4, -1.9824e-4, 4.372e-4 ];
    confInterval = 1.0;
    runFirst =  100;
  };

  graphtr =
  {
    type = "Graph";

    dataSets = ["train"];

    train =
    {
      key = "Convergence tr";

      yData = "userModules.solver.loss";
      xData = "userModules.solver.epoch";
    };
  };

  graphval =
  {
    type = "Graph";

    dataSets = "val";

    val =
    {
      key = "Convergence val";

      yData = "userModules.pred.error";
      xData = "userModules.pred.epoch";
    };
  };
};
