<!-- README.md is generated from README.Rmd. Please edit that file -->

# DexterMST

DexterMST is an R package acting as a companion to dexter and adding
facilities to manage and analyze data from multistage tests (MST). It
includes functions for importing and managing test data, assessing and
improving the quality of data through basic test and item analysis, and
fitting an IRT model, all adapted to the peculiarities of MST designs.
DexterMST typically works with project database files saved on disk.

## Installation

``` r
install.packages('dexterMST')
```

If you encounter a bug, please post a minimal reproducible example on
[github](https://github.com/dexter-psychometrics/dexter/issues). We post
news and examples on a
[blog](https://dexter-psychometrics.github.io/dexter/articles/blog/index.html),
it’s also the place for general questions.

## Example

Here is an example for a simple two-stage test.

``` r
library(dexterMST)
library(dplyr)
# start a project
db = create_mst_project(":memory:")

items = data.frame(item_id=sprintf("item%02i",1:70), item_score=1, delta=sort(runif(70,-1,1)))

design = data.frame(item_id=sprintf("item%02i",1:70),
                    module_id=rep(c('M4','M2','M5','M1','M6','M3', 'M7'),each=10))

routing_rules = routing_rules = mst_rules(
 `124` = M1[0:5] --+ M2[0:10] --+ M4, 
 `125` = M1[0:5] --+ M2[11:15] --+ M5,
 `136` = M1[6:10] --+ M3[6:15] --+ M6,
 `137` = M1[6:10] --+ M3[16:20] --+ M7)


scoring_rules = data.frame(
  item_id = rep(items$item_id,2), 
  item_score= rep(0:1,each=nrow(items)),
  response= rep(0:1,each=nrow(items))) # dummy respons
  

db = create_mst_project(":memory:")
add_scoring_rules_mst(db, scoring_rules)

create_mst_test(db,
                test_design = design,
                routing_rules = routing_rules,
                test_id = 'sim_test',
                routing = "all")
```

We can now plot the design

``` r
# plot test designs for all tests in the project
design_plot(db)
```

We now simulate data:

``` r
theta = rnorm(3000)

dat = sim_mst(items, theta, design, routing_rules,'all')
dat$test_id='sim_test'
dat$response=dat$item_score

add_response_data_mst(db, dat)
```

``` r
# IRT, extended nominal response model
f = fit_enorm_mst(db)

head(f)
```

| item_id | item_score |       beta |   SE_beta |
|:--------|-----------:|-----------:|----------:|
| item01  |          1 | -1.0863339 | 0.0626345 |
| item02  |          1 | -0.9418913 | 0.0623325 |
| item03  |          1 | -0.9251972 | 0.0623113 |
| item04  |          1 | -0.8020044 | 0.0622434 |
| item05  |          1 | -0.9318730 | 0.0623195 |
| item06  |          1 | -0.7521299 | 0.0622601 |

``` r
# ability estimates per person
rsp_data = get_responses_mst(db)
abl = ability(rsp_data, parms = f)
head(abl)
```

| booklet_id | person_id | booklet_score |      theta |
|:-----------|:----------|--------------:|-----------:|
| 136        | 1         |            19 |  0.8404993 |
| 125        | 10        |            19 |  0.2563194 |
| 124        | 100       |             9 | -1.3259574 |
| 136        | 1000      |            19 |  0.8404993 |
| 136        | 1001      |            14 |  0.1491514 |
| 125        | 1002      |            18 |  0.1129540 |

``` r
# ability estimates without item Item01
abl2 = ability(rsp_data, parms = f, item_id != "item01")

# plausible values
pv = plausible_values(rsp_data, parms = f, nPV = 5)
head(pv)
```

| booklet_id | person_id | booklet_score |       PV1 |        PV2 |       PV3 |       PV4 |        PV5 |
|:-----------|:----------|--------------:|----------:|-----------:|----------:|----------:|-----------:|
| 136        | 1         |            19 | 0.3839768 |  0.8766796 | 1.2262293 | 0.8659529 |  1.3278310 |
| 136        | 1000      |            19 | 0.4315371 |  0.4050729 | 1.0509408 | 0.5545282 |  0.9193252 |
| 136        | 1001      |            14 | 0.2054455 |  0.1277102 | 0.4737698 | 0.2015225 |  0.3163110 |
| 136        | 1006      |            16 | 0.3514111 | -0.0298913 | 0.6811140 | 0.0550563 |  0.2982861 |
| 136        | 1008      |            14 | 0.0542015 |  0.0259179 | 0.4488831 | 0.4268412 |  0.6982703 |
| 136        | 1009      |            14 | 0.0898933 |  0.4605046 | 0.3446569 | 0.0130914 | -0.1554303 |

## Contributing

Contributions are welcome but please check with us first about what you
would like to contribute.
