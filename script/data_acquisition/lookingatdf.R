library(dplyr)
#library("tidyverse")
library("stringr")
library(writexl)
library(readxl)
library(tidyr)

rm(list = ls())

setwd("/Users/Wolfalev/Documents/Grad/UT-AUSTIN GRAD/Coursework/CUSS/JiRepo/npo_classifier/script/data_acquisition/")

df <- read.csv("/Users/Wolfalev/Documents/Grad/UT-AUSTIN GRAD/Coursework/CUSS/JiRepo/npo_classifier/script/data_acquisition/texas_20132018_envorgs.csv")
namelist <- read.csv("/Users/Wolfalev/Documents/Grad/UT-AUSTIN GRAD/Coursework/CUSS/JiRepo/npo_classifier/dataset/union_ein_name_list.csv")
# 
df <- df %>% subset(EIN %in% namelist$EIN)
# length(unique(df_sub$EIN))

## 1. Data Checking

length(unique(df$EIN)) #904 unique orgs
length(unique(df$YEAR))


df <- df %>% group_by(EIN) %>% arrange(Real_Year, .by_group=TRUE)
class(df$Real_Year)
df_2000 <- df %>% subset(Real_Year == 2000)

df <- df %>% subset(Real_Year > 2012)

df_2018 <- df %>% subset(Real_Year==2018)
df_2017 <- df %>% subset(Real_Year==2017)
df_2016 <- df %>% subset(Real_Year==2016)
df_2015 <- df %>% subset(Real_Year==2015)
df_2014 <- df %>% subset(Real_Year==2014)
df_2013 <- df %>% subset(Real_Year==2013)

length(unique(df_2018$EIN))
length(unique(df_2017$EIN))
length(unique(df_2016$EIN))
length(unique(df_2015$EIN))
length(unique(df_2014$EIN))
length(unique(df_2013$EIN))

# df$year_id <- paste0(df$EIN,"_",df$Real_Year)
# length(unique(df$year_id))
# 
# df_2017 <- df_2017 %>% group_by(EIN) %>% mutate(n=n())
# 
# df_2017_test <- df_2017 %>% subset(n >1)

## 2. Data Checking

table(df$RETURN_TYPE)

## 990 Organizations 
df <- df %>% mutate(CombinedText = ifelse(RETURN_TYPE=="990"| RETURN_TYPE=="990O",paste0(IRS990_p1_ActvtyOrMssnDsc, " ",IRS990_p3_DscS," ",IRS990_p3_MssnDsc," ",IRS990ScheduleO_ExplntnTxt),
                                          ifelse(RETURN_TYPE=="990EZ"|RETURN_TYPE=="990EO", paste0(IRS990EZ_p3_DscrptnPrgrmSrvcAccmTxt, " ", IRS990EZ_p3_PrmryExmptPrpsTxt, " ", IRS990ScheduleO_ExplntnTxt),
                                                 ifelse(RETURN_TYPE=="990PF", paste0(IRS990PF_p9a_DscrptnTxt, " ",IRS990PF_p16b_RltnshpSttmntTxt), NA))))


df$CombinedText <-str_replace(df$CombinedText, "VERSION_NOT_SUPPORTED", "")
df$CombinedText <-str_replace(df$CombinedText, "VERSION_NOT_SUPPORTED", "")
df$CombinedText <-str_replace(df$CombinedText, "VERSION_NOT_SUPPORTED", "")
df$CombinedText <-str_replace(df$CombinedText, "VERSION_NOT_SUPPORTED", "")


table(is.na(df$CombinedText))
#FALSE  TRUE 
#1689  1128 

df <- df %>% group_by(EIN) %>% arrange(Real_Year, .by_group = TRUE) %>% fill(CombinedText) %>% fill(CombinedText, .direction="up" )

table(is.na(df$CombinedText))
#FALSE  TRUE 
#1804  1013 

table(df$Real_Year)

write.csv(df,"texas_20132018_envorgs_matched.csv")

