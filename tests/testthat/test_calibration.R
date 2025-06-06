
library(dplyr)

RcppArmadillo::armadillo_throttle_cores(1)
# 
# 
# items = data.frame(item_id=sprintf("item%02i",1:70), item_score=1, delta=sort(runif(70,-1,1)))
# 
# save(items,file=test_path('simdb','simitems.RData'))
# 
# make_sim_all = function()
# {
#   persons = tibble(person_id=1:3000,theta=rnorm(3000))
#   scoring_rules = data.frame(item_id=rep(paste0("item",sprintf("%02i",1:70)), each=2),
#                              response=rep(0:1,times=70),
#                               item_score=rep(0:1,times=70))
#   
#   design = data.frame(item_id=paste0("item",sprintf("%02i",1:70)),
#                       module_id=rep(c('M4','M2','M5','M1','M6','M3', 'M7'),times=rep(10,7)))
# 
#   unlink(test_path('simdb','simall.db'))
#   db = create_mst_project(test_path('simdb','simall.db'))
#   add_scoring_rules_mst(db, scoring_rules)
#   
#   add_item_properties_mst(db,select(items,-item_score))
#   
#   
#   routing_rules = mst_rules(
#     '124' = M1[0:5] --+ M2[0:10] --+ M4, 
#     '125' = M1[0:5] --+ M2[11:20] --+ M5,
#     '136' = M1[6:10] --+ M3[6:15] --+ M6,
#     '137' = M1[6:10] --+ M3[16:20] --+ M7)
#   
#   create_mst_test(db,
#                   test_design = design,
#                   routing_rules = routing_rules,
#                   test_id = 'EN',
#                   routing = "all")
# 
#   dat = sim_mst(items, persons$theta, design, routing_rules,'all')
#   dat$test_id='EN'
#   dat$response=dat$item_score
#   
#   add_response_data_mst(db, dat)
#   add_person_properties_mst(db,persons)
#   
#   close_mst_project(db)
# }
# 
# 
# make_sim_last = function()
# {
#   persons = tibble(person_id=1:3000,theta=rnorm(3000))
#   
#   scoring_rules = data.frame(item_id=rep(paste0("item",sprintf("%02i",1:70)), each=2),
#                              response=rep(0:1,times=70),
#                              item_score=rep(0:1,times=70))
#   
#   design = data.frame(item_id=paste0("item",sprintf("%02i",1:70)),
#                       module_id=rep(c('M4','M2','M5','M1','M6','M3', 'M7'),times=rep(10,7)))
#   
#   routing_rules = mst_rules(
#   '124' = M1[0:5] --+ M2[0:5] --+ M4, 
#   '125' = M1[0:5] --+ M2[6:10] --+ M5,
#   '136' = M1[6:10] --+ M3[0:5] --+ M6,
#   '137' = M1[6:10] --+ M3[6:10] --+ M7)
#   
#   unlink(test_path('simdb','simlast.db'))
#   db = create_mst_project(test_path('simdb','simlast.db'))
#   add_scoring_rules_mst(db, scoring_rules)
#   add_item_properties_mst(db,select(items,-item_score))
#   
#   create_mst_test(db,
#                   test_design = design,
#                   routing_rules = routing_rules,
#                   test_id = 'EN',
#                   routing = "last")
#   
#   
#   
#   dat = sim_mst(items, persons$theta, design, routing_rules,'last')
#   dat$test_id='EN'
#   dat$response=dat$item_score
#   
#   add_response_data_mst(db, dat)
# 
#   add_person_properties_mst(db,persons)
#   close_mst_project(db)
# }
# 
# make_sim_all()
# make_sim_last()

test_that('we can calibrate', {

  load(test_path('simdb','simitems.RData'))    
  all_db = open_mst_project(test_path('simdb','simall.db'))
  last_db = open_mst_project(test_path('simdb','simlast.db'))
  
  # all/last lead to approx same results
  
  fall = fit_enorm_mst(all_db)
  flast = fit_enorm_mst(last_db)

  expect_lt(mean(abs(coef(fall)$beta - coef(flast)$beta)),
            mean(coef(flast)$SE_b+coef(fall)$SE_b),
            'mean difference all<->last < mean se')
  
  # close to true item parameters
  
  tst = get_items_mst(all_db) %>%
    inner_join(coef(fall), by='item_id') %>%
    mutate(beta=beta-mean(beta),delta=delta-mean(delta)) %>%
    summarise(d = mean(abs(delta - beta)), se=mean(SE_beta))
    
  expect_lt(tst$d,tst$se, 'calibration delta close to true delta')        
  
  # predicates
  
  fall1 = fit_enorm_mst(all_db, item_id!='item32')
  flast1 = fit_enorm_mst(last_db, item_id!='item32')
  

  coef(fall1) %>%
    inner_join(coef(fall), by=c('item_id', 'item_score')) %>%
    mutate(d=abs(beta.x-beta.y)) %>%
    pull(d) %>%
    mean() %>%
    expect_lt(.01, 'all routing, omit item without problems')
  
  coef(flast1) %>%
    inner_join(coef(flast), by=c('item_id', 'item_score')) %>%
    mutate(d=abs(beta.x-beta.y)) %>%
    pull(d) %>%
    mean() %>%
    expect_lt(.01, 'last routing, omit item without problems')
  

  
  # test fixed parameters
  
  fixed = items[31:33,] %>% 
    rename(beta=delta) %>%
    mutate(beta=beta+3)
  
  f=fit_enorm_mst(all_db,fixed_parameters=fixed)
  
  tst = coef(f) %>% 
    inner_join(items, by=c('item_id','item_score'))
  
  expect_lt(abs(mean(tst$beta-3-tst$delta)),0.02)
  

  close_mst_project(all_db)
  close_mst_project(last_db)
  

})

RcppArmadillo::armadillo_reset_cores()
