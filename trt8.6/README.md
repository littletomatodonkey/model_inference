# TRT 8.6 模型转换、推理代码 demo

* ImageNet1k的class id映射表：[https://deeplearning.cms.waikato.ac.nz/user-guide/class-maps/IMAGENET/](https://deeplearning.cms.waikato.ac.nz/user-guide/class-maps/IMAGENET/)



## polygraphy: 模型diff对比工具（onnx-trt）

运行下面的命令，

```bash
sh cmopare_onnx_trt_diff.sh
```

会输出下面的内容，可以看出diff还是非常小的。

```
[I]     Comparing Output: '/conv1/Conv_output_0' (dtype=float32, shape=(1, 64, 112, 112)) with '/conv1/Conv_output_0' (dtype=float32, shape=(1, 64, 112, 112))
[I]         Tolerance: [abs=0.001, rel=0.001] | Checking elemwise error
[I]         onnxrt-runner-N0-11/15/23-12:55:37: /conv1/Conv_output_0 | Stats: mean=0.17684, std-dev=0.33954, var=0.11528, median=0.23743, min=-1.0369 at (0, 3, 106, 49), max=1.4111 at (0, 24, 17, 57), avg-magnitude=0.31578
[V]             ---- Histogram ----
                Bin Range          |  Num Elems | Visualization
                (-1.04  , -0.792 ) |      11758 | #
                (-0.792 , -0.547 ) |      34386 | #####
                (-0.547 , -0.302 ) |      42709 | ######
                (-0.302 , -0.0577) |      44587 | ######
                (-0.0577, 0.187  ) |     222773 | ##################################
                (0.187  , 0.432  ) |     260534 | ########################################
                (0.432  , 0.677  ) |     160032 | ########################
                (0.677  , 0.922  ) |      25643 | ###
                (0.922  , 1.17   ) |        379 | 
                (1.17   , 1.41   ) |         15 | 
[I]         trt-runner-N0-11/15/23-12:55:37: /conv1/Conv_output_0 | Stats: mean=0.17684, std-dev=0.33954, var=0.11528, median=0.23743, min=-1.0369 at (0, 3, 106, 49), max=1.4111 at (0, 24, 17, 57), avg-magnitude=0.31578
[V]             ---- Histogram ----
                Bin Range          |  Num Elems | Visualization
                (-1.04  , -0.792 ) |      11758 | #
                (-0.792 , -0.547 ) |      34386 | #####
                (-0.547 , -0.302 ) |      42709 | ######
                (-0.302 , -0.0577) |      44587 | ######
                (-0.0577, 0.187  ) |     222773 | ##################################
                (0.187  , 0.432  ) |     260534 | ########################################
                (0.432  , 0.677  ) |     160032 | ########################
                (0.677  , 0.922  ) |      25643 | ###
                (0.922  , 1.17   ) |        379 | 
                (1.17   , 1.41   ) |         15 | 
[I]         Error Metrics: /conv1/Conv_output_0
[I]             Minimum Required Tolerance: elemwise error | [abs=5.9605e-07] OR [rel=0.10345] (requirements may be lower if both abs/rel tolerances are set)
[I]             Absolute Difference | Stats: mean=2.3708e-08, std-dev=3.2795e-08, var=1.0755e-15, median=1.4901e-08, min=0 at (0, 0, 0, 3), max=5.9605e-07 at (0, 24, 97, 8), avg-magnitude=2.3708e-08
[V]                 ---- Histogram ----
                    Bin Range            |  Num Elems | Visualization
                    (0       , 5.96e-08) |     639810 | ########################################
                    (5.96e-08, 1.19e-07) |     135240 | ########
                    (1.19e-07, 1.79e-07) |      23138 | #
                    (1.79e-07, 2.38e-07) |       3315 | 
                    (2.38e-07, 2.98e-07) |       1070 | 
                    (2.98e-07, 3.58e-07) |        138 | 
                    (3.58e-07, 4.17e-07) |         73 | 
                    (4.17e-07, 4.77e-07) |         10 | 
                    (4.77e-07, 5.36e-07) |         19 | 
                    (5.36e-07, 5.96e-07) |          3 | 
[I]             Relative Difference | Stats: mean=3.8506e-07, std-dev=0.00011771, var=1.3856e-08, median=6.2326e-08, min=0 at (0, 0, 0, 3), max=0.10345 at (0, 24, 26, 33), avg-magnitude=3.8506e-07
[V]                 ---- Histogram ----
                    Bin Range        |  Num Elems | Visualization
                    (0     , 0.0103) |     802814 | ########################################
                    (0.0103, 0.0207) |          1 | 
                    (0.0207, 0.031 ) |          0 | 
                    (0.031 , 0.0414) |          0 | 
                    (0.0414, 0.0517) |          0 | 
                    (0.0517, 0.0621) |          0 | 
                    (0.0621, 0.0724) |          0 | 
                    (0.0724, 0.0828) |          0 | 
                    (0.0828, 0.0931) |          0 | 
                    (0.0931, 0.103 ) |          1 | 
[I]         PASSED | Output: '/conv1/Conv_output_0' | Difference is within tolerance (rel=0.001, abs=0.001)[I]     Comparing Output: '/conv1/Conv_output_0' (dtype=float32, shape=(1, 64, 112, 112)) with '/conv1/Conv_output_0' (dtype=float32, shape=(1, 64, 112, 112))
[I]         Tolerance: [abs=0.001, rel=0.001] | Checking elemwise error
[I]         onnxrt-runner-N0-11/15/23-12:55:37: /conv1/Conv_output_0 | Stats: mean=0.17684, std-dev=0.33954, var=0.11528, median=0.23743, min=-1.0369 at (0, 3, 106, 49), max=1.4111 at (0, 24, 17, 57), avg-magnitude=0.31578
[V]             ---- Histogram ----
                Bin Range          |  Num Elems | Visualization
                (-1.04  , -0.792 ) |      11758 | #
                (-0.792 , -0.547 ) |      34386 | #####
                (-0.547 , -0.302 ) |      42709 | ######
                (-0.302 , -0.0577) |      44587 | ######
                (-0.0577, 0.187  ) |     222773 | ##################################
                (0.187  , 0.432  ) |     260534 | ########################################
                (0.432  , 0.677  ) |     160032 | ########################
                (0.677  , 0.922  ) |      25643 | ###
                (0.922  , 1.17   ) |        379 | 
                (1.17   , 1.41   ) |         15 | 
[I]         trt-runner-N0-11/15/23-12:55:37: /conv1/Conv_output_0 | Stats: mean=0.17684, std-dev=0.33954, var=0.11528, median=0.23743, min=-1.0369 at (0, 3, 106, 49), max=1.4111 at (0, 24, 17, 57), avg-magnitude=0.31578
[V]             ---- Histogram ----
                Bin Range          |  Num Elems | Visualization
                (-1.04  , -0.792 ) |      11758 | #
                (-0.792 , -0.547 ) |      34386 | #####
                (-0.547 , -0.302 ) |      42709 | ######
                (-0.302 , -0.0577) |      44587 | ######
                (-0.0577, 0.187  ) |     222773 | ##################################
                (0.187  , 0.432  ) |     260534 | ########################################
                (0.432  , 0.677  ) |     160032 | ########################
                (0.677  , 0.922  ) |      25643 | ###
                (0.922  , 1.17   ) |        379 | 
                (1.17   , 1.41   ) |         15 | 
[I]         Error Metrics: /conv1/Conv_output_0
[I]             Minimum Required Tolerance: elemwise error | [abs=5.9605e-07] OR [rel=0.10345] (requirements may be lower if both abs/rel tolerances are set)
[I]             Absolute Difference | Stats: mean=2.3708e-08, std-dev=3.2795e-08, var=1.0755e-15, median=1.4901e-08, min=0 at (0, 0, 0, 3), max=5.9605e-07 at (0, 24, 97, 8), avg-magnitude=2.3708e-08
[V]                 ---- Histogram ----
                    Bin Range            |  Num Elems | Visualization
                    (0       , 5.96e-08) |     639810 | ########################################
                    (5.96e-08, 1.19e-07) |     135240 | ########
                    (1.19e-07, 1.79e-07) |      23138 | #
                    (1.79e-07, 2.38e-07) |       3315 | 
                    (2.38e-07, 2.98e-07) |       1070 | 
                    (2.98e-07, 3.58e-07) |        138 | 
                    (3.58e-07, 4.17e-07) |         73 | 
                    (4.17e-07, 4.77e-07) |         10 | 
                    (4.77e-07, 5.36e-07) |         19 | 
                    (5.36e-07, 5.96e-07) |          3 | 
[I]             Relative Difference | Stats: mean=3.8506e-07, std-dev=0.00011771, var=1.3856e-08, median=6.2326e-08, min=0 at (0, 0, 0, 3), max=0.10345 at (0, 24, 26, 33), avg-magnitude=3.8506e-07
[V]                 ---- Histogram ----
                    Bin Range        |  Num Elems | Visualization
                    (0     , 0.0103) |     802814 | ########################################
                    (0.0103, 0.0207) |          1 | 
                    (0.0207, 0.031 ) |          0 | 
                    (0.031 , 0.0414) |          0 | 
                    (0.0414, 0.0517) |          0 | 
                    (0.0517, 0.0621) |          0 | 
                    (0.0621, 0.0724) |          0 | 
                    (0.0724, 0.0828) |          0 | 
                    (0.0828, 0.0931) |          0 | 
                    (0.0931, 0.103 ) |          1 | 
[I]         PASSED | Output: '/conv1/Conv_output_0' | Difference is within tolerance (rel=0.001, abs=0.001)[I]     Comparing Output: '/conv1/Conv_output_0' (dtype=float32, shape=(1, 64, 112, 112)) with '/conv1/Conv_output_0' (dtype=float32, shape=(1, 64, 112, 112))
[I]         Tolerance: [abs=0.001, rel=0.001] | Checking elemwise error
[I]         onnxrt-runner-N0-11/15/23-12:55:37: /conv1/Conv_output_0 | Stats: mean=0.17684, std-dev=0.33954, var=0.11528, median=0.23743, min=-1.0369 at (0, 3, 106, 49), max=1.4111 at (0, 24, 17, 57), avg-magnitude=0.31578
[V]             ---- Histogram ----
                Bin Range          |  Num Elems | Visualization
                (-1.04  , -0.792 ) |      11758 | #
                (-0.792 , -0.547 ) |      34386 | #####
                (-0.547 , -0.302 ) |      42709 | ######
                (-0.302 , -0.0577) |      44587 | ######
                (-0.0577, 0.187  ) |     222773 | ##################################
                (0.187  , 0.432  ) |     260534 | ########################################
                (0.432  , 0.677  ) |     160032 | ########################
                (0.677  , 0.922  ) |      25643 | ###
                (0.922  , 1.17   ) |        379 | 
                (1.17   , 1.41   ) |         15 | 
[I]         trt-runner-N0-11/15/23-12:55:37: /conv1/Conv_output_0 | Stats: mean=0.17684, std-dev=0.33954, var=0.11528, median=0.23743, min=-1.0369 at (0, 3, 106, 49), max=1.4111 at (0, 24, 17, 57), avg-magnitude=0.31578
[V]             ---- Histogram ----
                Bin Range          |  Num Elems | Visualization
                (-1.04  , -0.792 ) |      11758 | #
                (-0.792 , -0.547 ) |      34386 | #####
                (-0.547 , -0.302 ) |      42709 | ######
                (-0.302 , -0.0577) |      44587 | ######
                (-0.0577, 0.187  ) |     222773 | ##################################
                (0.187  , 0.432  ) |     260534 | ########################################
                (0.432  , 0.677  ) |     160032 | ########################
                (0.677  , 0.922  ) |      25643 | ###
                (0.922  , 1.17   ) |        379 | 
                (1.17   , 1.41   ) |         15 | 
[I]         Error Metrics: /conv1/Conv_output_0
[I]             Minimum Required Tolerance: elemwise error | [abs=5.9605e-07] OR [rel=0.10345] (requirements may be lower if both abs/rel tolerances are set)
[I]             Absolute Difference | Stats: mean=2.3708e-08, std-dev=3.2795e-08, var=1.0755e-15, median=1.4901e-08, min=0 at (0, 0, 0, 3), max=5.9605e-07 at (0, 24, 97, 8), avg-magnitude=2.3708e-08
[V]                 ---- Histogram ----
                    Bin Range            |  Num Elems | Visualization
                    (0       , 5.96e-08) |     639810 | ########################################
                    (5.96e-08, 1.19e-07) |     135240 | ########
                    (1.19e-07, 1.79e-07) |      23138 | #
                    (1.79e-07, 2.38e-07) |       3315 | 
                    (2.38e-07, 2.98e-07) |       1070 | 
                    (2.98e-07, 3.58e-07) |        138 | 
                    (3.58e-07, 4.17e-07) |         73 | 
                    (4.17e-07, 4.77e-07) |         10 | 
                    (4.77e-07, 5.36e-07) |         19 | 
                    (5.36e-07, 5.96e-07) |          3 | 
[I]             Relative Difference | Stats: mean=3.8506e-07, std-dev=0.00011771, var=1.3856e-08, median=6.2326e-08, min=0 at (0, 0, 0, 3), max=0.10345 at (0, 24, 26, 33), avg-magnitude=3.8506e-07
[V]                 ---- Histogram ----
                    Bin Range        |  Num Elems | Visualization
                    (0     , 0.0103) |     802814 | ########################################
                    (0.0103, 0.0207) |          1 | 
                    (0.0207, 0.031 ) |          0 | 
                    (0.031 , 0.0414) |          0 | 
                    (0.0414, 0.0517) |          0 | 
                    (0.0517, 0.0621) |          0 | 
                    (0.0621, 0.0724) |          0 | 
                    (0.0724, 0.0828) |          0 | 
                    (0.0828, 0.0931) |          0 | 
                    (0.0931, 0.103 ) |          1 | 
[I]         PASSED | Output: '/conv1/Conv_output_0' | Difference is within tolerance (rel=0.001, abs=0.001)
```
