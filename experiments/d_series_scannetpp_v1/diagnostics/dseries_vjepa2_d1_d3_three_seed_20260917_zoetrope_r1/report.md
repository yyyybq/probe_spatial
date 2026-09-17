# V-JEPA2 D1/D3 three-seed development report

**development-only — no held-out test split; comparison signs are reported without an assumed direction.**

## D3 A/B/C/D comparisons

Differences are lhs minus rhs. MAE/AbsRel below zero and Delta1 above zero favour the lhs; confidence intervals are paired 69-scene cluster bootstraps after averaging the three seeds within each scene.

| branch | h | comparison | MAE delta [95% CI] | AbsRel delta [95% CI] | Delta1 delta [95% CI] | MAE seed SD |
|---|---:|---|---:|---:|---:|---:|
| pooled | 2 | B-A | 0.050618 [0.042440, 0.058855] | 0.063874 [0.052891, 0.075699] | -0.120728 [-0.139166, -0.102310] | 0.007597 |
| pooled | 2 | C-B | -0.009992 [-0.013729, -0.006173] | -0.012904 [-0.017262, -0.008397] | 0.025112 [0.015347, 0.034419] | 0.005184 |
| pooled | 2 | D-B | -0.015054 [-0.020551, -0.009521] | -0.019392 [-0.026058, -0.012860] | 0.032911 [0.019222, 0.046000] | 0.007372 |
| pooled | 2 | D-prior | -0.014214 [-0.020059, -0.008379] | -0.024965 [-0.032124, -0.017842] | 0.024278 [0.008358, 0.040045] | 0.000732 |
| pooled | 4 | B-A | 0.060782 [0.052801, 0.068882] | 0.072356 [0.062610, 0.082308] | -0.155795 [-0.176574, -0.135228] | 0.006650 |
| pooled | 4 | C-B | -0.010933 [-0.015539, -0.006529] | -0.012895 [-0.018459, -0.007530] | 0.026817 [0.016396, 0.037530] | 0.005726 |
| pooled | 4 | D-B | -0.014277 [-0.020412, -0.008450] | -0.016863 [-0.024548, -0.009430] | 0.034242 [0.019051, 0.049996] | 0.008098 |
| pooled | 4 | D-prior | -0.009730 [-0.015462, -0.004444] | -0.019059 [-0.026041, -0.012650] | 0.018514 [0.002994, 0.034938] | 0.000633 |
| spatial | 2 | B-A | 0.063452 [0.057570, 0.069374] | 0.067918 [0.060069, 0.075624] | -0.150583 [-0.164932, -0.136419] | 0.005600 |
| spatial | 2 | C-B | 0.009239 [0.007542, 0.010924] | 0.001679 [-0.000647, 0.003973] | -0.027321 [-0.033555, -0.020976] | 0.005610 |
| spatial | 2 | D-B | -0.022605 [-0.027389, -0.017950] | -0.018239 [-0.024319, -0.012185] | 0.055044 [0.041743, 0.068625] | 0.004968 |
| spatial | 2 | D-prior | -0.017344 [-0.022530, -0.012211] | -0.029189 [-0.035851, -0.022627] | 0.032833 [0.019804, 0.046183] | 0.000824 |
| spatial | 4 | B-A | 0.059929 [0.053359, 0.066653] | 0.063639 [0.055355, 0.072295] | -0.153180 [-0.168996, -0.137942] | 0.005075 |
| spatial | 4 | C-B | 0.007960 [0.005715, 0.010219] | -0.000609 [-0.003556, 0.002295] | -0.025033 [-0.033128, -0.017157] | 0.007051 |
| spatial | 4 | D-B | -0.015574 [-0.020939, -0.010614] | -0.010585 [-0.016792, -0.004841] | 0.037884 [0.023811, 0.052786] | 0.005187 |
| spatial | 4 | D-prior | -0.010580 [-0.015690, -0.005812] | -0.022349 [-0.028762, -0.016412] | 0.017693 [0.004049, 0.032052] | 0.000489 |

## Across-seed metric summary

| experiment | reference | branch | h | condition | MAE mean±SD | AbsRel mean±SD | Delta1 mean±SD |
|---|---|---|---:|---|---:|---:|---:|
| D1 | isolated_target | pooled | 2 | across_scene_shuffled_target | 0.296718±0.002224 | 0.337332±0.002160 | 0.459857±0.003336 |
| D1 | isolated_target | pooled | 2 | last_context | 0.363655±0.025840 | 0.359660±0.015962 | 0.327243±0.040193 |
| D1 | isolated_target | pooled | 2 | random | 0.497375±0.074384 | 0.496574±0.067359 | 0.231893±0.056751 |
| D1 | isolated_target | pooled | 2 | spatial_token_shuffle | 0.167153±0.001288 | 0.185856±0.002012 | 0.727185±0.005069 |
| D1 | isolated_target | pooled | 2 | train_layout_prior | 0.240631±0.000000 | 0.282667±0.000000 | 0.558110±0.000000 |
| D1 | isolated_target | pooled | 2 | true_target | 0.167603±0.001092 | 0.186237±0.001599 | 0.725441±0.004872 |
| D1 | isolated_target | pooled | 2 | within_scene_wrong_target | 0.283339±0.001922 | 0.320907±0.002027 | 0.475386±0.002597 |
| D1 | isolated_target | pooled | 2 | zero | 0.409353±0.063943 | 0.391246±0.067097 | 0.274945±0.051644 |
| D1 | isolated_target | pooled | 4 | across_scene_shuffled_target | 0.292447±0.002581 | 0.329201±0.002857 | 0.454972±0.003690 |
| D1 | isolated_target | pooled | 4 | last_context | 0.362221±0.027970 | 0.355749±0.020028 | 0.327604±0.038457 |
| D1 | isolated_target | pooled | 4 | random | 0.488104±0.049810 | 0.484395±0.039439 | 0.234583±0.030060 |
| D1 | isolated_target | pooled | 4 | spatial_token_shuffle | 0.166392±0.001860 | 0.182175±0.003007 | 0.727439±0.006446 |
| D1 | isolated_target | pooled | 4 | train_layout_prior | 0.237674±0.000000 | 0.275413±0.000000 | 0.551539±0.000000 |
| D1 | isolated_target | pooled | 4 | true_target | 0.166046±0.002116 | 0.181944±0.003350 | 0.727596±0.006533 |
| D1 | isolated_target | pooled | 4 | within_scene_wrong_target | 0.287299±0.003509 | 0.320029±0.003849 | 0.464097±0.005068 |
| D1 | isolated_target | pooled | 4 | zero | 0.409638±0.066652 | 0.391451±0.070650 | 0.274705±0.056119 |
| D1 | isolated_target | spatial | 2 | across_scene_shuffled_target | 0.291258±0.000393 | 0.330878±0.001733 | 0.462550±0.001289 |
| D1 | isolated_target | spatial | 2 | last_context | 0.318991±0.020712 | 0.348391±0.037113 | 0.420100±0.010841 |
| D1 | isolated_target | spatial | 2 | random | 0.672670±0.078291 | 0.668174±0.068935 | 0.105622±0.050065 |
| D1 | isolated_target | spatial | 2 | spatial_token_shuffle | 0.282413±0.000123 | 0.324610±0.002272 | 0.491858±0.001786 |
| D1 | isolated_target | spatial | 2 | train_layout_prior | 0.240631±0.000000 | 0.282667±0.000000 | 0.558110±0.000000 |
| D1 | isolated_target | spatial | 2 | true_target | 0.181321±0.000577 | 0.201958±0.002802 | 0.692333±0.000496 |
| D1 | isolated_target | spatial | 2 | within_scene_wrong_target | 0.278934±0.000786 | 0.315296±0.001177 | 0.481933±0.003553 |
| D1 | isolated_target | spatial | 2 | zero | 0.290380±0.020887 | 0.283755±0.008418 | 0.416679±0.051995 |
| D1 | isolated_target | spatial | 4 | across_scene_shuffled_target | 0.281976±0.000369 | 0.318079±0.001704 | 0.472733±0.001951 |
| D1 | isolated_target | spatial | 4 | last_context | 0.321033±0.020709 | 0.349200±0.036766 | 0.417832±0.008108 |
| D1 | isolated_target | spatial | 4 | random | 0.672285±0.080365 | 0.668135±0.071911 | 0.105347±0.051079 |
| D1 | isolated_target | spatial | 4 | spatial_token_shuffle | 0.274017±0.000469 | 0.312237±0.001987 | 0.497195±0.001657 |
| D1 | isolated_target | spatial | 4 | train_layout_prior | 0.237674±0.000000 | 0.275413±0.000000 | 0.551539±0.000000 |
| D1 | isolated_target | spatial | 4 | true_target | 0.177945±0.000441 | 0.196204±0.002264 | 0.697446±0.000760 |
| D1 | isolated_target | spatial | 4 | within_scene_wrong_target | 0.276819±0.000434 | 0.308868±0.001828 | 0.478484±0.002050 |
| D1 | isolated_target | spatial | 4 | zero | 0.284080±0.021418 | 0.274930±0.009628 | 0.424033±0.047749 |
| D1 | native_target | pooled | 2 | across_scene_shuffled_target | 0.287951±0.004308 | 0.327879±0.003265 | 0.466738±0.010129 |
| D1 | native_target | pooled | 2 | last_context | 0.275070±0.006022 | 0.309417±0.005216 | 0.496532±0.015791 |
| D1 | native_target | pooled | 2 | random | 0.453730±0.034884 | 0.463455±0.030432 | 0.278613±0.039468 |
| D1 | native_target | pooled | 2 | spatial_token_shuffle | 0.190851±0.001262 | 0.213215±0.003296 | 0.670206±0.003242 |
| D1 | native_target | pooled | 2 | train_layout_prior | 0.240631±0.000000 | 0.282667±0.000000 | 0.558110±0.000000 |
| D1 | native_target | pooled | 2 | true_target | 0.190854±0.001262 | 0.213219±0.003296 | 0.670205±0.003250 |
| D1 | native_target | pooled | 2 | within_scene_wrong_target | 0.274066±0.003140 | 0.310452±0.003554 | 0.495588±0.004041 |
| D1 | native_target | pooled | 2 | zero | 0.403191±0.055206 | 0.386819±0.055844 | 0.288376±0.048732 |
| D1 | native_target | pooled | 4 | across_scene_shuffled_target | 0.280278±0.004304 | 0.317558±0.003798 | 0.474072±0.006769 |
| D1 | native_target | pooled | 4 | last_context | 0.276023±0.004119 | 0.309185±0.003413 | 0.488341±0.009301 |
| D1 | native_target | pooled | 4 | random | 0.458192±0.038768 | 0.464621±0.030090 | 0.262378±0.038653 |
| D1 | native_target | pooled | 4 | spatial_token_shuffle | 0.181436±0.001955 | 0.200857±0.003059 | 0.691618±0.005676 |
| D1 | native_target | pooled | 4 | train_layout_prior | 0.237674±0.000000 | 0.275413±0.000000 | 0.551539±0.000000 |
| D1 | native_target | pooled | 4 | true_target | 0.181440±0.001956 | 0.200861±0.003060 | 0.691606±0.005687 |
| D1 | native_target | pooled | 4 | within_scene_wrong_target | 0.279321±0.004463 | 0.313074±0.004104 | 0.474131±0.007897 |
| D1 | native_target | pooled | 4 | zero | 0.403675±0.056688 | 0.387132±0.058192 | 0.285568±0.051584 |
| D1 | native_target | spatial | 2 | across_scene_shuffled_target | 0.289577±0.002473 | 0.329373±0.004190 | 0.466403±0.003109 |
| D1 | native_target | spatial | 2 | last_context | 0.288985±0.004370 | 0.296109±0.005008 | 0.444899±0.006105 |
| D1 | native_target | spatial | 2 | random | 0.645763±0.043372 | 0.640244±0.037342 | 0.115820±0.031185 |
| D1 | native_target | spatial | 2 | spatial_token_shuffle | 0.284282±0.001439 | 0.326953±0.003700 | 0.484428±0.002440 |
| D1 | native_target | spatial | 2 | train_layout_prior | 0.240631±0.000000 | 0.282667±0.000000 | 0.558110±0.000000 |
| D1 | native_target | spatial | 2 | true_target | 0.182441±0.000288 | 0.203799±0.001962 | 0.686481±0.001830 |
| D1 | native_target | spatial | 2 | within_scene_wrong_target | 0.280151±0.002697 | 0.318348±0.003792 | 0.481035±0.004035 |
| D1 | native_target | spatial | 2 | zero | 0.327422±0.021324 | 0.304163±0.013851 | 0.335417±0.042600 |
| D1 | native_target | spatial | 4 | across_scene_shuffled_target | 0.276172±0.000863 | 0.311080±0.001321 | 0.479277±0.000435 |
| D1 | native_target | spatial | 4 | last_context | 0.297400±0.004130 | 0.303169±0.004412 | 0.429414±0.005562 |
| D1 | native_target | spatial | 4 | random | 0.643116±0.045662 | 0.638021±0.039268 | 0.116432±0.033159 |
| D1 | native_target | spatial | 4 | spatial_token_shuffle | 0.272709±0.000604 | 0.309743±0.001746 | 0.495459±0.001177 |
| D1 | native_target | spatial | 4 | train_layout_prior | 0.237674±0.000000 | 0.275413±0.000000 | 0.551539±0.000000 |
| D1 | native_target | spatial | 4 | true_target | 0.182739±0.000330 | 0.200009±0.001729 | 0.684528±0.001350 |
| D1 | native_target | spatial | 4 | within_scene_wrong_target | 0.272959±0.000886 | 0.303899±0.001166 | 0.483044±0.002198 |
| D1 | native_target | spatial | 4 | zero | 0.322074±0.021871 | 0.297345±0.014941 | 0.344519±0.043954 |
| D3 | native_target | pooled | 2 | frozen_readout_transfer | 0.241471±0.006658 | 0.277093±0.005824 | 0.549478±0.013895 |
| D3 | native_target | pooled | 2 | last_context | 0.275070±0.006022 | 0.309417±0.005216 | 0.496532±0.015791 |
| D3 | native_target | pooled | 2 | native_target_reference | 0.190854±0.001262 | 0.213219±0.003296 | 0.670205±0.003250 |
| D3 | native_target | pooled | 2 | retrained_same_capacity_readout | 0.226417±0.000732 | 0.257701±0.001983 | 0.582389±0.001138 |
| D3 | native_target | pooled | 2 | tokenwise_linear_adapter | 0.231480±0.001772 | 0.264189±0.002010 | 0.574590±0.003548 |
| D3 | native_target | pooled | 2 | train_layout_prior | 0.240631±0.000000 | 0.282667±0.000000 | 0.558110±0.000000 |
| D3 | native_target | pooled | 2 | within_scene_wrong_target | 0.274066±0.003140 | 0.310452±0.003554 | 0.495588±0.004041 |
| D3 | native_target | pooled | 4 | frozen_readout_transfer | 0.242222±0.007510 | 0.273217±0.006234 | 0.535811±0.017512 |
| D3 | native_target | pooled | 4 | last_context | 0.276023±0.004119 | 0.309185±0.003413 | 0.488341±0.009301 |
| D3 | native_target | pooled | 4 | native_target_reference | 0.181440±0.001956 | 0.200861±0.003060 | 0.691606±0.005687 |
| D3 | native_target | pooled | 4 | retrained_same_capacity_readout | 0.227945±0.000633 | 0.256354±0.001304 | 0.570053±0.004453 |
| D3 | native_target | pooled | 4 | tokenwise_linear_adapter | 0.231289±0.001803 | 0.260322±0.003246 | 0.562628±0.005653 |
| D3 | native_target | pooled | 4 | train_layout_prior | 0.237674±0.000000 | 0.275413±0.000000 | 0.551539±0.000000 |
| D3 | native_target | pooled | 4 | within_scene_wrong_target | 0.279321±0.004463 | 0.313074±0.004104 | 0.474131±0.007897 |
| D3 | native_target | spatial | 2 | frozen_readout_transfer | 0.245893±0.005787 | 0.271716±0.003988 | 0.535899±0.016911 |
| D3 | native_target | spatial | 2 | last_context | 0.288985±0.004370 | 0.296109±0.005008 | 0.444899±0.006105 |
| D3 | native_target | spatial | 2 | native_target_reference | 0.182441±0.000288 | 0.203799±0.001962 | 0.686481±0.001830 |
| D3 | native_target | spatial | 2 | retrained_same_capacity_readout | 0.223288±0.000824 | 0.253478±0.002760 | 0.590943±0.001688 |
| D3 | native_target | spatial | 2 | tokenwise_linear_adapter | 0.255132±0.011281 | 0.273395±0.005077 | 0.508578±0.034095 |
| D3 | native_target | spatial | 2 | train_layout_prior | 0.240631±0.000000 | 0.282667±0.000000 | 0.558110±0.000000 |
| D3 | native_target | spatial | 2 | within_scene_wrong_target | 0.280151±0.002697 | 0.318348±0.003792 | 0.481035±0.004035 |
| D3 | native_target | spatial | 4 | frozen_readout_transfer | 0.242669±0.005394 | 0.263649±0.004743 | 0.531348±0.017344 |
| D3 | native_target | spatial | 4 | last_context | 0.297400±0.004130 | 0.303169±0.004412 | 0.429414±0.005562 |
| D3 | native_target | spatial | 4 | native_target_reference | 0.182739±0.000330 | 0.200009±0.001729 | 0.684528±0.001350 |
| D3 | native_target | spatial | 4 | retrained_same_capacity_readout | 0.227094±0.000489 | 0.253064±0.001559 | 0.569232±0.002003 |
| D3 | native_target | spatial | 4 | tokenwise_linear_adapter | 0.250629±0.011123 | 0.263039±0.003133 | 0.506315±0.035807 |
| D3 | native_target | spatial | 4 | train_layout_prior | 0.237674±0.000000 | 0.275413±0.000000 | 0.551539±0.000000 |
| D3 | native_target | spatial | 4 | within_scene_wrong_target | 0.272959±0.000886 | 0.303899±0.001166 | 0.483044±0.002198 |
