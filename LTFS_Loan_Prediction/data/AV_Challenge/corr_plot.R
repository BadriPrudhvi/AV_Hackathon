library(dplyr)
library(ggcorrplot)

train_file = read.csv('~/PycharmProjects/fastai/courses/ml1/data/AV_Challenge/clean_train.csv'
                      ,stringsAsFactors = FALSE)
truncated_df = train_file[,c(1:39,41:43)]
truncated_df <- truncated_df %>% select(-MobileNo_Avl_Flag)
nums <- unlist(lapply(truncated_df, is.numeric))  

ggcorrplot(corr = cor(truncated_df[,nums])
           ,show.legend = TRUE
           ,tl.srt = 90
           , type =c('lower')
           )

corr_df = cor(truncated_df[,nums]) %>% data.frame()