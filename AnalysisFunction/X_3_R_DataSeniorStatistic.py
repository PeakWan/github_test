# AnalysisFunction.
# V 1.0.13
# date 2021-5-26
# author：xuyuan

# V1.0.0 更新说明： 初始logistic 和liner 回归
# V1.0.1 更新说明： 初始COX 回归
# V1.0.2 更新说明：修改了交互效应的连接符号
# V1.0.3 更新说明：所有方法的列线图增加trycatch，cox增加timeroc入参
# V1.0.4 更新说明：新增了R_surv_ana
# V1.0.5 更新说明：增加了列线图宽度，所有方法增加参数style
# V1.0.5 更新说明：森林图去除截距，优化生存分析，新增NRI分析
# V1.0.6 更新说明：生存分析增加风格参数
# V1.0.7 更新说明：优化了生存分析和NRI的小bug
# v1.0.8 更新说明： 优化初始logistic和cox ，logist 增加返回{error: } ,cox新增参数calibrate
# v1.0.9 更新说明： logist 增加返回{error: }
# v1.0.10 更新说明：cox 增加参数 u
# v1.0.11 更新说明：cox 建模计算参数更换为“breslow”
# v1.0.12 更新说明：cox 优化，新增入参 timepreinc,timeprelist
# v1.0.12 更新说明：logist\liner\cox增加定类变量控制
# v1.0.13 更新说明：更新R_logistic_regression（更新有序多分类算法）
#v 1.0.14 更新说明：增加R_RCS_ana算法

import pandas as pd
import rpy2.robjects as ro
from rpy2.robjects.packages import importr
from rpy2.robjects import pandas2ri
from rpy2.robjects.conversion import localconverter
import AnalysisFunction.X_5_R_SmartPlot as x5r
from AnalysisFunction.X_3_DataSeniorStatistics import get_variables
from AnalysisFunction.X_3_DataSeniorStatistics import variables_control
import re
import datetime
import numpy as np

# import time
# import timeout_decorator

# @timeout_decorator.timeout(5)
def R_logistic_regression(df_input, dependent_variable, continuous_independent_variables,
                          categorical_independent_variables,
                          categorical_independent_variables_ref, interaction_effects_variables, savePath,
                          model_name='logit',
                          step_method='all', decimal_num=3, style=1):
    """
    logistic回归
    df_input：DataFrame 输入的待处理数据
    dependent_variable：str 应变量
    continuous_independent_variables：list 定量自变量
    categorical_independent_variables:list 定类自变量
    categorical_independent_variables_ref:list 定类自变量 参考标签 长度需要与定类自变量一致
    interaction_effects_variables:list 交互效应
    savePath:str 图片路径
    model_name:str 模型名
    step_method:str 逐步回归方法 forward、backward、both、all
    decimal_num: int  小数点位数
    """

    LogReg = ro.r('''
#V1.0.5
#Author shanqingsong
#date 2020-12-18

#V1.0.2：第一版logistics回归模型
#V1.0.3：更改转变factor变量的方式
#V1.0.4：增加描述
#V1.0.5：增加有序多分类方法,最多支持5个有序类
LogReg <- function(mydata,target, feature_con,feature_cat, interaction = NULL,cate=NULL, cat_ref_lev=NULL,method="ALL", nomogram = "yes",mod='logit',savePath,round= 2){
  # input:
  # mydata:dataframe 需处理的数据
  # target:str 应变量
  # feature_con: strVector 定量因变量
  # feature_cat: strVector 定类因变量
  # interaction: strVector 交互作用因变量
  # cate: strVector 需要设置参考标签的定类因变量
  # cat_ref_lev: strVector 定类因变量参考标签
  # method:str 逐步回归方法 forward、backward、both、all
  # nomogram: str 是否生成列线图 yes no 有交互效应时无法使用
  # mod: str logist还是多分类
  # savePath:str 图片保存路径
  # round:int 小数点位数
  # return:
  # results$coefs:dataframe 表格结果
  # results$roc: strVector ROC图片路径
  # results$nomo: strVector 列线图片路径
  # results$data: dataframe 预测结果
  # results$descrip:strVector 描述结果
  library(MASS)
  library(pROC)
  library(rms)
  library(dplyr)
  results <- list() #结果
  mytime <- format(Sys.time(), "%b_%d_%H_%M_%S_%Y")  #时间
  rand <- sample(1:100,1)
  # 数据预处理
  origdata <- mydata
  for (cat_ in feature_cat){
    mydata[,cat_]=as.factor(mydata[,cat_])
  }
  #mydata[,feature_cat] <- lapply(mydata[,feature_cat], factor)
  mydata <- na.omit(mydata) # 除去NA
  if(length(cate)>0){
    for (i in 1:length(cate)){
      mydata[,cate[i]] <-  relevel(mydata[,cate[i]], ref = cat_ref_lev[i])
    }
  }
  # feature_con 连续变量
  # feature_cat 类别变量
  # interaction 交互项
  # cate 需要指定参考标签的类别变量
  feature <- c(feature_con,feature_cat, interaction)
  # 建立模型
  formula <- paste0(target,'~',paste(feature,collapse = '+'))
  # logistic 模型
  if(mod=='logit'){
    fit <- glm(as.formula(formula),data=mydata,family = binomial())
    # 变量选择。 outputfit为输出模型
    # 将下面改为switch
    if (method == "forward"){
      outputfit <- step(fit,direction = 'forward')
    } else
      if (method == "backward"){
        outputfit <- step(fit,direction = 'backward')
      } else
        if(method == "both"){
          outputfit <- step(fit,direction = 'both')
        } else{
          outputfit <- fit
        }
    # 输出系数及or值的整理
    results$model <- outputfit
    summ <- summary(outputfit)
    #for(i in 1:nrow(summ$coefficients)){
    # print(paste0(rownames(summ$coefficients)[i],'变量的系数为',round(summ$coefficients[i,1],round),',p值为',round(summ$coefficients[i,4],round)))
    #if (summ$coefficients[i,4]<=0.05){
    #results$descrip <-paste0(results$descrip  ,paste0(rownames(summ$coefficients)[i],'变量的系数为',round(summ$coefficients[i,1],round),',p值为',round(summ$coefficients[i,4],round)))
    #}
    #}
    pvalue <- summary(outputfit)$coefficient[,4]
    coefs <- cbind(coef(outputfit), confint(outputfit), exp(coef(outputfit)), exp(confint(outputfit)),pvalue)
    coefs <- round(coefs,round)
    rownames(coefs)  <- NULL
    colnames(coefs) <- c("系数","系数下限","系数上限","OR","OR下限","OR上限","pvalue")
    coefs <- data.frame(name = names(outputfit$coefficients), coefs)
    #print(coef)
    results$coefs <- coefs
  }else{#否则就是有序多分类
  # 输出系数及or值的整理
  mydata[,which(colnames(mydata)==target)] <- as.ordered(mydata[,which(colnames(mydata)==target)])
  outputfit <- polr(formula,data=mydata,Hess = TRUE)
  results$model <- outputfit
  summ <- summary(outputfit)   
  ctable <- coef(summ)
  pvalue <- pnorm(abs(ctable[, "t value"]), lower.tail = FALSE) * 2
  coefs <- cbind(coef(outputfit), confint(outputfit), exp(coef(outputfit)), exp(confint(outputfit)),pvalue[1:length(coef(outputfit))])
  coefs <- round(coefs,round)
  rownames(coefs)  <- NULL
  colnames(coefs) <- c("系数","系数下限","系数上限","OR","OR下限","OR上限","pvalue")
  coefs <- data.frame(name = names(outputfit$coefficients), coefs)
  #print(coefs)
  results$coefs <- coefs
  }
  
  # 画nomogram， 想法是从最优模型中得到formula，代入lrm命令中
  if((nomogram=="yes")&(mod=='logit')){
    tryCatch({
      ddist <<- datadist(mydata)
      options(datadist='ddist')
      tempfit <- lrm(formula(outputfit),data=mydata)
      print(tempfit)
      nom <- nomogram(tempfit, fun=plogis,
                      fun.at=c(seq(.1,.9, by=.1)),
                      lp=F, funlabel="Risk")
      Nomoname <- paste0("Nomogram",mytime,"_",rand,".jpeg")
      jpeg(file=paste0(savePath,Nomoname),width=7,height=8,units="in",res=600)
      plot(nom)
      dev.off()
      results$nomo <- Nomoname
    },error=function(e){
      print('变量系数过大，无法生成列线图。')
    })
  }
  if((nomogram=="yes")&(mod!='logit')){
    tryCatch({
      ddist <<- datadist(mydata)
      options(datadist='ddist')
      modord <- lrm(formula(outputfit),data=mydata)
      if(length(table(mydata[target]))==3){
        fun2 <- function(x) plogis(x-modord$coef[1]+modord$coef[2])#3个类
        funlist=list('Prob Y>=1'=plogis,'Prob Y>=2'=fun2)
      }else if(length(table(mydata[target]))==4){
        fun2 <- function(x) plogis(x-modord$coef[1]+modord$coef[2])#3个类
        fun3 <- function(x) plogis(x-modord$coef[1]+modord$coef[3])#4个类
        funlist=list('Prob Y>=1'=plogis,'Prob Y>=2'=fun2,'Prob Y>=3'=fun3)
      }else{
        fun2 <- function(x) plogis(x-modord$coef[1]+modord$coef[2])#3个类
        fun3 <- function(x) plogis(x-modord$coef[1]+modord$coef[3])#4个类
        fun4 <- function(x) plogis(x-modord$coef[1]+modord$coef[4])#5个类
        funlist=list('Prob Y>=1'=plogis,'Prob Y>=2'=fun2,'Prob Y>=3'=fun3,'Prob Y>=4'=fun4)
      }
      nom <- nomogram(modord, fun=funlist,
                      lp=F,
                      fun.at=c(.01,.05,seq(.1,.9,by=.1),.95,.99))
      Nomoname <- paste0("Nomogram",mytime,"_",rand,".jpeg")
      jpeg(file=paste0(savePath,Nomoname),width=7,height=8,units="in",res=600)
      plot(nom)
      dev.off()
      results$nomo <- Nomoname
    },error=function(e){
      print('变量系数过大，无法生成列线图。')
    })
  }
  # return(results)
  
  # 画ROC曲线
  if(mod=='logit'){
    gfit <- roc(as.formula(paste0(target,'~predict(outputfit)')), data = mydata)
    ROCname <- paste0("ROC",mytime,"_",rand,".jpeg")
    jpeg(file=paste0(savePath,ROCname),width=5,height=5,units="in",res=600)
    plot.roc(gfit,col="red",print.auc =TRUE,print.auc.col = "darkgreen",auc.polygon = TRUE,auc.polygon.col = "pink")
    dev.off()
    #输出ROC数据
    #xcoor <- seq(from = 0,to = 1,length.out = 50)
    #coordi <- coords(gfit, x = xcoor, transpose = FALSE)
    #results$coord <- data.frame("sensitivity" = coordi$sensitivity, "specificity" = coordi$specificity, "Youden" = coordi$sensitivity + coordi$specificity -1)
    #results$coord <- results$coord %>% round(round)
    results$roc <- ROCname
    # mydata %>% predict(outputfit, ., type="response") %>% print()
    # mydata %>% predict(outputfit, ., type="response") %>% head() %>% print()
    
  }
 #--------添加描述---------------
  if(mod=='logit'){
    coefstat <- NULL
    for(i in 2:nrow(summ$coefficients)){
      coefstat[i] <- paste0('变量',rownames(summ$coefficients)[i],'的系数为',round(summ$coefficients[i,1],round),',p值为',
                            round(summ$coefficients[i,4],round), ",",
                            case_when(
                              summ$coefficients[i,4]<0.05 ~ "是显著的。",
                              summ$coefficients[i,4]>=0.05 ~ "是不显著的。@"
                            )
      ) 
    }
    results$descrip <- paste0("本研究采用二分类Logistic回归评估", paste0(feature,collapse = "、"), "对",target,"的影响。",
                              "模型的AUC为", round(gfit$auc,2), ",",
                              case_when(
                                gfit$auc<0.5 ~ "请检查模型",
                                gfit$auc==0.5 ~ "模型没有预测价值",
                                gfit$auc>0.5 &  gfit$auc<=0.7 ~ "模型预测效果较差",
                                gfit$auc>0.7 &  gfit$auc<=0.85 ~ "模型预测效果一般",
                                gfit$auc>0.85~ "模型预测效果较好"
                              ),"。" ,"纳入模型的变量中，",  paste0(coefstat[2:nrow(summ$coefficients)],collapse = " ")
    )
  }else{#多分类的情况下的描述
    coefstat <- NULL
    for(i in 1:nrow(coefs)){
      coefstat[i] <- paste0('变量',coefs$name[i],'的系数为',round(coefs[i,2],round),',p值为',
                            round(coefs[i,8],round), ",",
                            case_when(
                              coefs[i,8]<0.05 ~ "是显著的。",
                              coefs[i,8]>=0.05 ~ "是不显著的。@"
                            )
      ) 
    }
    predictions=predict(outputfit,newdata = mydata,type='prob')
    muroc = multiclass.roc(mydata[,which(colnames(mydata)==target)], predictions)
    ##AUC可以计算                          判断target所在列号
    results$descrip <- paste0("本研究采用多分类Logistic回归评估", paste0(feature,collapse = "、"), "对",target,"的影响。",
                              "模型的AUC为", round(muroc$auc,2), ",",
                              case_when(
                                muroc$auc<0.5 ~ "请检查模型",
                                muroc$auc==0.5 ~ "模型没有预测价值",
                                muroc$auc>0.5 &  muroc$auc<=0.7 ~ "模型预测效果较差",
                                muroc$auc>0.7 &  muroc$auc<=0.85 ~ "模型预测效果一般",
                                muroc$auc>0.85~ "模型预测效果较好"
                              ),"。" ,"纳入模型的变量中，",  paste0(coefstat[1:nrow(coefs)],collapse = " ")
    )
  }
  
  #------添加预测结果----------
    if(mod=='logit'){
      results$data<- data.frame(mydata %>% predict(outputfit, ., type="response")  %>%  mutate(origdata,.))
    }else{
      results$data<- data.frame(mydata %>% predict(outputfit, ., type="prob")  %>%  mutate(origdata,.))
    }
  #这一句并没有实际对应输出,为什么没有看到forestplot但是有森林图？
  return(results)
}

            '''
                  )
    # 对照ref生成新的list
    temp_df = pd.DataFrame()
    temp_df['cat'] = categorical_independent_variables
    temp_df['cat_ref_lev'] = categorical_independent_variables_ref
    temp_df.dropna(inplace=True)
    cat_ = temp_df['cat'].astype(str)
    cat_ref_lev_ = temp_df['cat_ref_lev'].astype(str)
    df_origin = df_input
    df_input_features = [dependent_variable] + continuous_independent_variables + categorical_independent_variables
    if len(interaction_effects_variables) > 0:
        interaction_effects_variables_ = get_variables(interaction_effects_variables, "\*")
        df_input_features += interaction_effects_variables_
        df_input_features = list(set(df_input_features))
    df_input = df_input[df_input_features].dropna()
    #控制分类变量的分类数
    str_variables_control=variables_control(df_input, categorical_independent_variables, "logit")
    if str_variables_control:
        return {'error': str_variables_control}
    try:
        df_input[[dependent_variable]] = df_input[[dependent_variable]].astype(int)
    except Exception as e:
        print(e)
        return {'error': '应变量无法转为数值'}
    dv_unique = pd.unique(df_input[dependent_variable])
#    if set(dv_unique) != set([0, 1]):
#        return {'error': '应变量只允许取值为0、1的二分类数据。' + str(dependent_variable) + ':' + str(dv_unique)}
    if model_name == 'logit' and set(dv_unique) != set([0, 1]):
        return {'error': '二元logistics模型中应变量只允许取值为0、1的二分类数据。如要继续建模请选择其他模型' + str(dependent_variable) + ':' + str(dv_unique)}
    elif model_name =='mulclass'and len(dv_unique)>5:
        return {'error': '有序多分类模型中应变量类别数目最多5个。' + str(dependent_variable) + ':' + str(dv_unique)}

    with localconverter(ro.default_converter + ro.pandas2ri.converter):
        r_df = ro.conversion.py2rpy(df_input)
    target = dependent_variable
    feature_con = ro.StrVector(continuous_independent_variables)
    feature_cat = ro.StrVector(categorical_independent_variables)
    cat = ro.StrVector(cat_)
    cat_ref_lev = ro.StrVector(cat_ref_lev_)
    if len(interaction_effects_variables) > 0:
        interaction = ro.StrVector(interaction_effects_variables)
        result = LogReg(mydata=r_df, feature_con=feature_con, feature_cat=feature_cat, interaction=interaction,
                        cate=cat, cat_ref_lev=cat_ref_lev,
                        target=target, method=step_method, nomogram="no",mod=model_name, savePath=savePath, round=decimal_num)

    else:
        result = LogReg(mydata=r_df, feature_con=feature_con, feature_cat=feature_cat,
                        cate=cat, cat_ref_lev=cat_ref_lev,
                        target=target, method=step_method, nomogram="yes", mod=model_name,savePath=savePath, round=decimal_num)
    list_plot_path = []
    with localconverter(ro.default_converter + ro.pandas2ri.converter):
        df_pred_temp = ro.conversion.rpy2py(result.rx2('data'))
        df_result = ro.conversion.rpy2py(result.rx2('coefs'))
        str_result = tuple(result.rx2('descrip'))[0]
        if model_name =='logit':
            list_plot_path.append(tuple(result.rx2('roc'))[0])
        if len(interaction_effects_variables) == 0:
            try:
                list_plot_path.append(tuple(result.rx2('nomo'))[0])
            except Exception as e:
                print(e)
    df_result = df_result.reset_index(drop=True)
    df_pred_temp = df_pred_temp.reset_index(drop=True)
    #df_pred = pd.concat([df_origin, df_pred_temp[['.']]], axis=1)
    df_pred = pd.concat([df_origin, df_pred_temp.iloc[:,-1]], axis=1)
    pred_name = 'pred_' + str(datetime.datetime.now().hour) + str(datetime.datetime.now().minute) + str(
        datetime.datetime.now().second)
    #df_pred.rename(columns={".": pred_name}, inplace=True)
    #try:
    #    froest_plot_path = x5r.R_froest_plot(df_input=df_result.iloc[1:, :], title='OR(95%CI)', name='name', mean="OR",
    #                                         lowerb="OR下限", upperb="OR上限", savePath=savePath, grawid=3, tickfont=1,
    #                                         xlabfont=1, style=style, zero=1)
    #except Exception as e:
    #    print(e)
    #    froest_plot_path = ''
    #list_plot_path.append(froest_plot_path)
    str_result = str_result.replace('@', '\n')
    df_result.rename(columns={'OR下限': 'OR下限(95%CI)', 'OR上限': 'OR上限(95%CI)'}, inplace=True)
    return df_result, str_result, list_plot_path, df_pred


def R_liner_regression(df_input, dependent_variable, continuous_independent_variables,
                       categorical_independent_variables,
                       categorical_independent_variables_ref, interaction_effects_variables, savePath,
                       model_name='liner',
                       step_method='all', decimal_num=3, style=1):
    """
    线性回归
    df_input：DataFrame 输入的待处理数据
    dependent_variable：str 应变量
    continuous_independent_variables：list 定量自变量
    categorical_independent_variables:list 定类自变量
    categorical_independent_variables_ref:list 定类自变量 参考标签 长度需要与定类自变量一致
    interaction_effects_variables:list 交互效应
    savePath:str 图片路径
    model_name:str 模型名
    step_method:str 逐步回归方法 forward、backward、both、all
    decimal_num: int  小数点位数
    """

    Regfun = ro.r('''
            #V1.0.4
#Author shanqingsong
#date 2020-12-10

#V1.0.2：第一版logistics回归模型
#V1.0.3：更高转变factor变量的方式
#V1.0.4：新增描述
LinearReg <- function(mydata,target, feature_con,feature_cat, interaction = NULL,cate=NULL, cat_ref_lev=NULL,method="ALL", nomogram = "yes",savePath,round= 2){
# input:
      # mydata:dataframe 需处理的数据
      # target:str 应变量
      # feature_con: strVector 定量因变量
      # feature_cat: strVector 定类因变量
      # interaction: strVector 交互作用因变量
      # cate: strVector 需要设置参考标签的定类因变量
      # cat_ref_lev: strVector 定类因变量参考标签
      # method:str 逐步回归方法 forward、backward、both、all
      # nomogram: str 是否生成列线图 yes no 有交互效应时无法使用
      # savePath:str 图片保存路径
      # round:int 小数点位数
# return:
      # results$coefs:dataframe 表格结果
      # results$descrip:strVector 描述结果
      # results$nomo: strVector 列线图片路径
      # results$residualplot: strVector 残差图图片路径
      # results$qqplot: strVector qq图片路径
      # results$data: dataframe 预测结果

              library(MASS)
              library(pROC)
              library(rms)
              library(dplyr)
              results <- list() #结果
              mytime <- format(Sys.time(), "%b_%d_%H_%M_%S_%Y")  #时间
              rand <- sample(1:100,1)
              # 数据预处理
              origdata <- mydata
              for (cat_ in feature_cat){
                 mydata[,cat_]=as.factor(mydata[,cat_])
              }
              mydata <- na.omit(mydata) # 除去NA
              if(length(cate)>0){
                for (i in 1:length(cate)){
                  mydata[,cate[i]] <-  relevel(mydata[,cate[i]], ref = cat_ref_lev[i])
                }
              }
              # feature_con 连续变量
              # feature_cat 类别变量
              # interaction 交互项
              # cate 需要指定参考标签的类别变量
              feature <- c(feature_con,feature_cat, interaction)
              # 建立模型
              formula <- paste0(target,'~',paste(feature,collapse = '+'))
              # liner 模型
              fit <- glm(as.formula(formula),data=mydata,family = gaussian())
              # 变量选择。 outputfit为输出模型
              # 将下面改为switch
              if (method == "forward"){
                outputfit <- step(fit,direction = 'forward')
              } else
                if (method == "backward"){
                  outputfit <- step(fit,direction = 'backward')
                } else
                  if(method == "both"){
                    outputfit <- step(fit,direction = 'both')
                  } else{
                    outputfit <- fit
                  }
              # 输出系数及or值的整理
              results$model <- outputfit
              Residualplot <- paste0("Residualplot",mytime,"_",rand,".jpeg")
              jpeg(file=paste0(savePath,Residualplot),width=5,height=5,units="in",res=600)
              plot(outputfit, which= 1)
              dev.off()
              QQplot <- paste0("QQplot",mytime,"_",rand,".jpeg")
              jpeg(file=paste0(savePath,QQplot),width=5,height=5,units="in",res=600)
              plot(outputfit, which= 2)
              dev.off()
              summ <- summary(outputfit)
              # for(i in 1:nrow(summ$coefficients)){
              #    # print(paste0(rownames(summ$coefficients)[i],'变量的系数为',round(summ$coefficients[i,1],round),',p值为',round(summ$coefficients[i,4],round)))
              #    if (summ$coefficients[i,4]<=0.05){
              #     results$descrip <-paste0(results$descrip  ,paste0(rownames(summ$coefficients)[i],'变量的系数为',round(summ$coefficients[i,1],round),',p值为',round(summ$coefficients[i,4],round)))
              #    }
              # }
              pvalue <- summary(outputfit)$coefficient[,4]
              coefs <- cbind(coef(outputfit), confint(outputfit),pvalue)
              coefs <- round(coefs,round)
              rownames(coefs)  <- NULL
              colnames(coefs) <- c("系数","系数下限","系数上限","pvalue")
              coefs <- data.frame(name = names(outputfit$coefficients), coefs)
              results$coefs <- coefs
              #--------增加描述------------
              coefstat <- NULL
              for(i in 2:nrow(summ$coefficients)){
                coefstat[i] <- paste0('变量',rownames(summ$coefficients)[i],'的系数为',round(summ$coefficients[i,1],round),',p值为',
                                      round(summ$coefficients[i,4],round), ",",
                                      case_when(
                                        summ$coefficients[i,4]<0.05 ~ "是显著的。",
                                        summ$coefficients[i,4]>=0.05 ~ "是不显著的。"
                                      )
                ) 
              }
              results$descrip <- paste0("本研究采用广义线性回归模型评估", paste0(feature,collapse = "、"), "对",target,"的影响。",
                                           "模型的AIC为", round(summ$aic,2), "。",
                                           "纳入模型的变量中，",  paste0(coefstat[2:3],collapse = " ")
              ) 
              # 画nomogram， 想法是从最优模型中得到formula，代入lrm命令中
              if(nomogram=="yes"){
                tryCatch({
                ddist <<- datadist(mydata)
                options(datadist='ddist')
                tempfit <- lrm(formula(outputfit),data=mydata)
                #print(tempfit)
                nom <- nomogram(tempfit, fun=plogis,
                                fun.at=c(seq(.1,.9, by=.1)),
                                lp=F, funlabel="Risk")
                Nomoname <- paste0("Nomogram",mytime,"_",rand,".jpeg")
                jpeg(file=paste0(savePath,Nomoname),width=7,height=8,units="in",res=600)
                plot(nom)
                dev.off()
                results$nomo <- Nomoname
                 },error=function(e){
                    print('变量系数过大，无法生成列线图。')
                    })
              }
              results$residualplot <- Residualplot
              results$qqplot <- QQplot
              # mydata %>% predict(outputfit, ., type="response") %>% print()
              # mydata %>% predict(outputfit, ., type="response") %>% head() %>% print()
              results$data<- (mydata %>% predict(outputfit, ., type="response")  %>%  mutate(origdata,.))
              return(results)
            }
            '''
                  )
    # 对照ref生成新的list
    temp_df = pd.DataFrame()
    temp_df['cat'] = categorical_independent_variables
    temp_df['cat_ref_lev'] = categorical_independent_variables_ref
    temp_df.dropna(inplace=True)
    cat_ = temp_df['cat'].astype(str)
    cat_ref_lev_ = temp_df['cat_ref_lev'].astype(str)
    df_origin = df_input
    df_input_features = [dependent_variable] + continuous_independent_variables + categorical_independent_variables
    if len(interaction_effects_variables) > 0:
        interaction_effects_variables_ = get_variables(interaction_effects_variables, "\*")
        df_input_features += interaction_effects_variables_
        df_input_features = list(set(df_input_features))
    df_input = df_input[df_input_features].dropna()

    #控制分类变量的分类数
    str_variables_control=variables_control(df_input, categorical_independent_variables, "ols")
    if str_variables_control:
        return {'error': str_variables_control}

    with localconverter(ro.default_converter + ro.pandas2ri.converter):
        r_df = ro.conversion.py2rpy(df_input)
    target = dependent_variable
    feature_con = ro.StrVector(continuous_independent_variables)
    feature_cat = ro.StrVector(categorical_independent_variables)
    cat = ro.StrVector(cat_)
    cat_ref_lev = ro.StrVector(cat_ref_lev_)
    if len(interaction_effects_variables) > 0:
        interaction = ro.StrVector(interaction_effects_variables)
        result = Regfun(mydata=r_df, feature_con=feature_con, feature_cat=feature_cat, interaction=interaction,
                        cate=cat, cat_ref_lev=cat_ref_lev,
                        target=target, method=step_method, nomogram="no", savePath=savePath, round=decimal_num)
    else:
        result = Regfun(mydata=r_df, feature_con=feature_con, feature_cat=feature_cat,
                        cate=cat, cat_ref_lev=cat_ref_lev,
                        target=target, method=step_method, nomogram="yes", savePath=savePath, round=decimal_num)
    list_plot_path = []
    # print(tuple(result))
    with localconverter(ro.default_converter + ro.pandas2ri.converter):
        df_pred_temp = ro.conversion.rpy2py(result.rx2('data'))
        df_result = ro.conversion.rpy2py(result.rx2('coefs'))
        str_result = tuple(result.rx2('descrip'))[0]
        list_plot_path.append(tuple(result.rx2('residualplot'))[0])
        list_plot_path.append(tuple(result.rx2('qqplot'))[0])
        if len(interaction_effects_variables) == 0:
            try:
                list_plot_path.append(tuple(result.rx2('nomo'))[0])
            except Exception as e:
                print(e)
    df_result = df_result.reset_index(drop=True)
    df_pred_temp = df_pred_temp.reset_index(drop=True)
    df_pred = pd.concat([df_origin, df_pred_temp[['.']]], axis=1)
    pred_name = 'pred_' + str(datetime.datetime.now().hour) + str(datetime.datetime.now().minute) + str(
        datetime.datetime.now().second)
    df_pred.rename(columns={".": pred_name}, inplace=True)
    try:
        froest_plot_path = x5r.R_froest_plot(df_input=df_result.iloc[1:, :], title='β(95%CI)', name='name', mean="系数",
                                             lowerb="系数下限", upperb="系数上限", savePath=savePath, grawid=3, tickfont=1,
                                             xlabfont=1, style=style, zero=0)
    except Exception as e:
        print(e)
        froest_plot_path = ''

    list_plot_path.append(froest_plot_path)
    return df_result, str_result, list_plot_path, df_pred


def R_cox_regression(df_input, sta_variable, tim_variable, continuous_independent_variables,
                     categorical_independent_variables,
                     categorical_independent_variables_ref, interaction_effects_variables, savePath,
                     timequant=[0.2, 0.4, 0.6, 0.8],
                     step_method='all', timeroc='no', calibrate="no", decimal_num=3, style=1, u=None,
                     timepreinc='year', timeprelist=None):
    """
    cox回归
    df_input：DataFrame 输入的待处理数据
    sta_variable：str 状态变量
    tim_variable：str 时间变量
    continuous_independent_variables：list 定量自变量
    categorical_independent_variables:list 定类自变量
    categorical_independent_variables_ref:list 定类自变量 参考标签 长度需要与定类自变量一致
    interaction_effects_variables:list 交互效应
    savePath:str 图片路径
    timequant:list 时间分位点
    step_method:str 逐步回归方法 forward、backward、both、all
    timeroc:是否展示时间依赖ROC no yes
    calibrate:是否展示校准曲线 no yes
    decimal_num: int  小数点位数
    u: int 模型预测时间点
    timepreinc: str  列线图以年月日哪个为单位 year month day
    timeprelist: list  生成3条以内的生存概率线，每条代表的时间
    """

    Coxfun = ro.r('''
        #V1.0.3
        #Author yuanke
        #date 2021-05-7

        #V1.0.0：第一版COX回归模型
        #V1.0.1: 新增校准曲线
        #v1.0.2: 新增u参数进行列线图和校准曲线优化
        #v1.0.3: 新增timepreinc时间单位和timeprelist时间长度
        CoxReg <- function(mydata,tim,sta, feature_con=NULL,feature_cat=NULL, 
                           interaction = NULL,cate=NULL, cat_ref_lev=NULL,method='all',
                           nomogram = "no",timepreinc='year',timeprelist=NULL,calibrate="no", timeroc="no",
                           timequant=NULL, savePath,round= 2,u=NULL){
          # input:
          # mydata:dataframe 需处理的数据
          # sta: 生存状态，0：删失，1：生存，2：死亡
          # target:str 应变量
          # feature_con: strVector 定量因变量
          # feature_cat: strVector 定类因变量
          # strata_var: 分层变量(只在绘制生存曲线时用)
          # interaction: strVector 交互作用因变量
          # cate: strVector 需要设置参考标签的定类因变量
          # cat_ref_lev: strVector 定类因变量参考标签
          # nomogram: str 是否生成列线图 yes no
          # timepreinc:str  列线图以年月日哪个为单位 year month day
          # timeprelist:   生成3条以内的生存概率线，每条代表的时间
          #                全部折算为days时,当设置的days小于一定值时因为观测时间太短而不能出图
          # calibrate: str 是否生成校准曲线 yes no
          # timeroc:是否展示时间依赖ROC no yes
          # timequant: 时间分位点
          # savePath:str 图片保存路径
          # round:int 小数点位数
          # u:决定列线图和校准曲线优化参数
          # return:
          # results$coefs:dataframe 表格结果
          # results$TimeROCname: strVector 分段roc
          # results$TimeAUCname: strVector 连续roc
          # results$survname: strVector 生存图
          # results$nomo: strVector 列线图片路径
          # results$conclusion: strVector 描述
          library(MASS)
          library(pROC)
          library(rms)
          library(dplyr)
          library(survival)
          library(survminer)
          library(timeROC)

          results <- list() #结果
          mytime <- format(Sys.time(), "%b_%d_%H_%M_%S_%Y")  #时间
          rand <- sample(1:100,1)
          #if (is.na(u)) u <- max(mydata$tim)
          # 数据预处理

          # for (cat_ in feature_cat){
          #   mydata[,cat_]=as.factor(mydata[,cat_])
          # }
          mydata %<>% mutate_at(feature_cat, funs(factor(.))) %>% as.data.frame()
          # cate 需包含在feature_cat内
          if(length(cate)>0){
            for (i in 1:length(cate)){
              mydata[,cate[i]] <-  relevel(mydata[,cate[i]], ref = cat_ref_lev[i])
            }
          }

          feature1 <- c(feature_con,feature_cat, interaction)
          # 建立模型

          formula <- paste0("Surv","(",tim,",",sta,")",'~',paste(feature1,collapse = '+'))
          print(formula)
          fit <- coxph(as.formula(formula), data = mydata,ties = "breslow")

          #逐步回归
          if (method == "forward"){
            fit <- step(fit,direction = 'forward')
          } else
            if (method == "backward"){
              fit <- step(fit,direction = 'backward')
            } else
              if(method == "both"){
                fit <- step(fit,direction = 'both')
              } else{
                fit <- fit
              }
          results$model <- fit
          # 绘制生存曲线
          Survname <- paste0("Survgram",mytime,".jpeg")
          #若模型中没有有意义的变量
          if (is.null(coef(fit))){
            formula <- paste("Surv","(",tim,",",sta,")","~",1)
            fit <- coxph(as.formula(formula), data=mydata,ties = "breslow")
            results$model <- fit
            ggsurvplot(survfit(fit), data = mydata, palette = "#2E9FDF", 
                       ggtheme = theme_minimal(), legend = "none")
            ggsave(filename = Survname, path = savePath)
            results$survname<-Survname
            results$conclusion <-'模型中没有纳入有意义的变量'
            return(results)
          }
          ggsurvplot(survfit(fit), data = mydata, palette = "#2E9FDF", 
                     ggtheme = theme_minimal(), legend = "none")
          ggsave(filename = Survname, path = savePath)
          results$survname = Survname

          # 列线图
          if(nomogram=="yes"){
            ddist <<- datadist(mydata)
            options(datadist='ddist')
            #print(fit$formula)
            tempfit <- cph(fit$formula,data=mydata,x=T,y=T,surv = T)
            surv<-Survival(tempfit)
            surv1<-function(x)surv(timeprelist[1],x)
            flist<-list(surv1)
            if(length(timeprelist)==2){
              surv2<-function(x)surv(timeprelist[2],x)
              flist<-list(surv1,surv2)
            }else if(length(timeprelist)==3){
              surv2<-function(x)surv(timeprelist[2],x)
              surv3<-function(x)surv(timeprelist[3],x)
              flist<-list(surv1,surv2,surv3)
            }

            funlab <- list()
            for (i in (1:length(timeprelist))){
              funlab[i]<-paste0(timeprelist[i],'-',timepreinc,' survival')
            }
            fl <- as.character(funlab)

            nom <- nomogram(tempfit, fun=flist,
                            fun.at=c(seq(.1,.9, by=.1)),
                            funlabel=fl)
            Nomoname <- paste0("Nomogram",mytime,"_",rand,".jpeg")
            jpeg(file=paste0(savePath,Nomoname),width=10,height=8,units="in",res=600)
            plot(nom,xfrac=.4)
            dev.off()
            results$nomo <- Nomoname
          }

          # 时间依赖ROC分段
          if(timeroc=="yes"){
            if(!is.null(timequant)){
                mydata$lp <- predict(fit, type ="lp")
                if(mean(timequant)<1){
                    timedep <- quantile(mydata[,tim],probs=timequant)
                }else
                if(mean(timequant)>1){
                    timedep <- timequant 
                }# 下接ROC.marginal
              ROC.marginal<-timeROC(T=mydata[,tim],
                                    marker = mydata$lp,
                                    delta=mydata[,sta],
                                    cause=1,weighting="marginal",
                                    times= timedep,
                                    iid=TRUE)
              TimeROCname <- paste0("TimeROC",mytime,"_",rand,".jpeg")
              jpeg(file=paste0(savePath,TimeROCname),width=5,height=5,units="in",res=600)
              plot(ROC.marginal,time=timedep[1], col= palette()[1], title= FALSE)
              for(i in 2:length(timedep)){
                plot(ROC.marginal,time=timedep[i],add=TRUE, col = i)
              }
              legend <- paste0(timedep, rep("(AUC = ", length(ROC.marginal$AUC)), round(ROC.marginal$AUC,2), rep(")", length(ROC.marginal$AUC)))
              legend(x="bottomright",  legend = legend, col = 1:length(timedep),lty=1)
              dev.off()
              results$TimeROCname = TimeROCname
              # 时间依赖ROC连续
              TimeAUCname <- paste0("TimeAUC",mytime,"_",rand,".jpeg")
              jpeg(file=paste0(savePath,TimeAUCname),width=5,height=5,units="in",res=600)
              plotAUCcurve(ROC.marginal,conf.int=TRUE,conf.band=TRUE)
              dev.off()
              results$TimeAUCname = TimeAUCname
            }
          }
          #生成coef
          pvalue <- summary(fit)$coefficient[,5]
          coefs <- cbind(coef(fit), exp(coef(fit)), exp(confint(fit)),pvalue)
          coefs <- round(coefs,round)
          rownames(coefs)  <- NULL
          colnames(coefs) <- c("系数","HR","HR下限(95%CI)","HR上限(95%CI)","P")
          results$coef  <- data.frame(name = names(fit$coefficients), coefs)
          summ <- summary(fit)

          # return(summ)
          coefstat <- NULL
          for(i in 1:nrow(summ$coefficients)){
            if(summ$coefficients[i,5]<0.05){
              coefstat[i] <- paste0('变量',rownames(summ$coefficients)[i],'的系数为',round(summ$coefficients[i,1],round),
                                    ',HR值为', round(summ$coefficients[i,2],round),
                                    case_when(
                                      round(summ$coefficients[i,2],round)<1 ~ paste0("说明其将阳性事件的发生风险降低了",
                                                                                     (1-round(summ$coefficients[i,2],round))*100,"%"),
                                      round(summ$coefficients[i,2],round)==1 ~ "说明其对阳性事件的发生风险无影响",
                                      round(summ$coefficients[i,2],round)>1 ~ paste0("说明其将阳性事件的发生风险提高了",
                                                                                     round(summ$coefficients[i,2]-1,round)*100,"%")
                                    ),
                                    ',p值为',
                                    round(summ$coefficients[i,5],round), ",",
                                    case_when(
                                      summ$coefficients[i,5]<0.05 ~ "是显著的，",
                                      summ$coefficients[i,5]>=0.05 ~ "是不显著的，"
                                    ), "95%置信区间为 (",round(summ$conf.int[i,3],round),round(summ$conf.int[i,4],round), ")",'@'
              )
            }else{coefstat[i] <- paste0('变量',rownames(summ$coefficients)[i],'不显著','@')}
          }
          # print(coefstat)
          results$conclusion <- paste0("本研究采用Cox比例风险模型评估", paste0(feature1,collapse = "、"), "对生存期的影响。",
                                       "其中总样本数为",nrow(mydata),",死亡个体数为",summ$nevent, "。", "模型wald检验P值为",round(summ$waldtest[3],round),
                                       case_when(
                                         summ$waldtest[3]<0.05 ~ "是显著的。",
                                         summ$waldtest[3]>=0.05 ~ "是不显著的。"
                                       ),
                                       "模型logrank检验P值为",round(summ$sctest[3],round),
                                       case_when(
                                         summ$sctest[3]<0.05 ~ "是显著的。",
                                         summ$sctest[3]>=0.05 ~ "是不显著的。"
                                       ),
                                       "模型似然比检验P值为",round(summ$logtest[3],round),
                                       case_when(
                                         summ$logtest[3]<0.05 ~ "是显著的。",
                                         summ$logtest[3]>=0.05 ~ "是不显著的。"
                                       ),'@',
                                       "模型的C-index即一致性指数(concordance index)为",
                                       round(summ$concordance[1],round),"。@",
                                       "纳入模型的变量中，",  paste0(coefstat[1:nrow(summ$coefficients)],collapse = " ")
          )
          #校准曲线
          if (calibrate=="yes"){
            rk <- sort(mydata$tim)
            v <- floor(0.95*nrow(mydata))
            v <- rk[v]
            if (u>v) u <- v                  
            msg <- tryCatch({
              picname <- paste0("jzqx",mytime,"_",rand,".jpeg")
              jpeg(file=paste0(savePath,picname),width=10,height=10,units="in",res=600)  
              coxm<- cph(as.formula(formula),data=mydata,surv=T,x=T,y=T,time.inc = u)
              cal<-calibrate(coxm,u=u,cmethod='KM',m=nrow(mydata)/5, B=100)
              xl<-paste0('Nomogram Predicated Probability of ',u,'-',timepreinc)
              yl<-paste0('Actual rate of ',u,'-',timepreinc)
              plot(cal,xlim = c(0,1),ylim= c(0,1),
                   errbar.col=c(rgb(0,0,0,maxColorValue=255)),
                   col=c(rgb(255,0,0,maxColorValue=255)),
                   xlab = xl, 
                   ylab = yl)
              abline(0,1,lty=1,lwd=2,col=c(rgb(0,0,255,maxColorValue= 255)))
              dev.off()
              results$picjzqx<- picname
            }, error = function(e) {
              "error"
              dev.off()
            })
            if(msg=="error"){
              results$conclusion <- paste0(results$conclusion,
                                           "无法生成校准曲线，设置的预测时间点过大")
            }

          }
          return(results)
         }
    ''')
    # 对照ref生成新的list
    temp_df = pd.DataFrame()
    temp_df['cat'] = categorical_independent_variables
    temp_df['cat_ref_lev'] = categorical_independent_variables_ref
    temp_df.dropna(inplace=True)
    cat_ = temp_df['cat'].astype(str)
    cat_ref_lev_ = temp_df['cat_ref_lev'].astype(str)

    df_input_features = [sta_variable,
                         tim_variable] + continuous_independent_variables + categorical_independent_variables
    if len(interaction_effects_variables) > 0:
        interaction_effects_variables_ = get_variables(interaction_effects_variables, "\*")
        df_input_features += interaction_effects_variables_
        df_input_features = list(set(df_input_features))
    df_input = df_input[df_input_features].dropna()

    #控制分类变量的分类数
    str_variables_control=variables_control(df_input, categorical_independent_variables, "cox")
    if str_variables_control:
        return {'error': str_variables_control}

    if u is None:
        u = int(np.median(df_input[tim_variable]))
    try:
        df_input[[sta_variable]] = df_input[[sta_variable]].astype(int)
    except Exception as e:
        print(e)

    if timeprelist is None:
        timeprelist = [u]

    with localconverter(ro.default_converter + ro.pandas2ri.converter):
        r_df = ro.conversion.py2rpy(df_input)
    feature_con = ro.StrVector(continuous_independent_variables)
    feature_cat = ro.StrVector(categorical_independent_variables)
    cat = ro.StrVector(cat_)
    cat_ref_lev = ro.StrVector(cat_ref_lev_)
    timequant = ro.FloatVector(timequant)
    timeprelist = ro.FloatVector(timeprelist)
    # print(timequant)

    if len(interaction_effects_variables) > 0:
        interaction = ro.StrVector(interaction_effects_variables)
        result = Coxfun(mydata=r_df, feature_con=feature_con, feature_cat=feature_cat, interaction=interaction,
                        cate=cat, cat_ref_lev=cat_ref_lev,
                        tim=tim_variable, sta=sta_variable, method=step_method, nomogram="no", timequant=timequant,
                        timeroc=timeroc, calibrate=calibrate, savePath=savePath, round=decimal_num, u=u,
                        timepreinc=timepreinc, timeprelist=timeprelist)
    else:
        result = Coxfun(mydata=r_df, feature_con=feature_con, feature_cat=feature_cat,
                        cate=cat, cat_ref_lev=cat_ref_lev,
                        tim=tim_variable, sta=sta_variable, method=step_method, nomogram="yes", timequant=timequant,
                        timeroc=timeroc, calibrate=calibrate, savePath=savePath, round=decimal_num, u=u,
                        timepreinc=timepreinc, timeprelist=timeprelist)
    list_plot_path = []
    # print(tuple(result))
    with localconverter(ro.default_converter + ro.pandas2ri.converter):
        try:
            df_result = ro.conversion.rpy2py(result.rx2('coef'))
        except:
            df_result = pd.DataFrame()
        str_result = tuple(result.rx2('conclusion'))[0]
        list_plot_path.append(tuple(result.rx2('survname'))[0])
        try:
            list_plot_path.append(tuple(result.rx2('TimeAUCname'))[0])
            list_plot_path.append(tuple(result.rx2('TimeROCname'))[0])
        except Exception as e:
            print(e)
        try:
            list_plot_path.append(tuple(result.rx2('picjzqx'))[0])
        except Exception as e:
            print(e)
        if len(interaction_effects_variables) == 0:
            try:
                list_plot_path.append(tuple(result.rx2('nomo'))[0])
            except Exception as e:
                print(e)
    # df_result = df_result.reset_index(drop=True)
    try:
        froest_plot_path = x5r.R_froest_plot(df_input=df_result, title='HR(95%CI)', name='name', mean="HR",
                                             lowerb="HR下限.95.CI.", upperb="HR上限.95.CI.", savePath=savePath, grawid=3,
                                             tickfont=1,
                                             xlabfont=1, style=style, zero=1)
        list_plot_path.append(froest_plot_path)
    except Exception as e:
        print(e)
    str_result = str_result.replace('@', '\n')
    df_result.rename(columns={'HR下限.95.CI.': 'HR下限(95%CI)', 'HR上限.95.CI.': 'HR上限(95%CI)'}, inplace=True)
    return df_result, str_result, list_plot_path


def R_surv_ana(df_input, time, statu, groups, methods, conf_int, risk_table, ncensor_plot, x_distance, path,
               decimal_num=3, palette_style='nejm'):
    """
    生存分析
    df_input：DataFrame 输入的待处理数据
    time：str 时间变量
    statu: str 状态变量
    groups:str：单个组别
    methods:str:  1=Kapian-Meier生存曲线,2=Cumulative hazard风险累积曲线
    conf_int:    1=输出置信区间，0=不输出
    risk_table:  1=输出risk表，0=不输出
    ncensor_plot：1=输出censor表，0=不输出
    x_distance:int 作图x轴的刻度间隔
    path:str 图片保存路径
    decimal_num: int  小数点位数
    palette_style:str 图片风格参数 （nejm、lancet、jama、npg）
    """
    surv_ana = ro.r('''
#V1.0.3
#Author yuanke
#date 2021-1-31

#V1.0.0:第一版生存分析
#v1.0.1:加入了曲线出图出表选择
#V1.0.2:增加全局变量
#V1.0.3:增加风格参数 （nejm,lancet,jama,bmj）
surv.ana <- function(df,tim_name, sta_name,group_name,method=NULL,
                     conf_int=NULL,risk_table=NULL,ncensor_plot=NULL,
                     xlab_distance=NULL, savePath,round=NULL,style='nejm'){
  # input:
  # df:dataframe 需处理的数据
  # timevariable:vector  时间变量
  # statusvariable: vector 状态变量
  #group:             vector：组别
  #method:str:  1=Kapian-Meier生存曲线,2=Cumulative hazard风险累积曲线
  #conf_int:    1=输出置信区间，0=不输出
  #risk_table:  1=输出risk表，0=不输出
  #ncensor_plot：1=输出censor表，0=不输出
  #xlab_distance:int 作图x轴的刻度间隔
  # savePath:str 图片保存路径
  # round:int 小数点位数
  # return:
  # results$picpath: 生存曲线or风险累积曲线保存路径
  # results$tb:dataframe   表格结果
  # results$descrip1:strVector 智能描述结果&Log Rank分析结果
  library(survival)
  library(survminer)
  library(dplyr)
  library(ggplot2)
  library(ggpubr)
  tim=as.numeric(df[,tim_name])
  sta=as.numeric(df[,sta_name])
  group=df[,group_name]
  tim<<-tim
  sta<<-sta
  group<<-group
  results <- list() #结果
  mytime <- format(Sys.time(), "%b_%d_%H_%M_%S_%Y")  #系统时间
  rand <- sample(1:100,1)
  x_lab=tim_name
  fit1<- survfit(Surv(tim,sta)~group,data=df)
  surv_diff <- survdiff(Surv(tim,sta)~group, data = df)
  pvalue <- 1-pchisq(surv_diff$chisq,length(surv_diff$n)-1)
  descrip=paste0("以group分组的患者的生存时间的Log Rank秩和检验统计量=",round(surv_diff$chisq,round),'，对应的p=',round(pvalue,round),'，按照Log Rank秩和检验结果,认为不同group的患者生存时间',
           case_when(
             pvalue<0.05 ~ "存在显著性差异。",
             pvalue>=0.05 ~ "无显著性差异。"
           )
           )
           table1=summary(fit1)$table#无法生成删失数和删失率
           #Log Rank
           #surv_diff <- survdiff(Surv(time, status) ~ group, data = df)放到最开始的ifelse
           #pvalue <- 1-pchisq(surv_diff$chisq,length(surv_diff$n)-1)#放到ifelse
           tb <- cbind(table1[,1], table1[,4],table1[,7],table1[,8],table1[,9])
           rownames(tb)  <- rownames(table1)
           colnames(tb) <- c("总数","事件数","中位数","95%LCI","95%UCI")
           results$table <- data.frame(tb)
           for(i in 1:nrow(table1)){
             descrip <- paste0(descrip,rownames(table1)[i],'的患者中status存活中位数为',round(tb[i,3],round),
                                   '存活50%对应的存活时间95%置信区间：','[',round(tb[i,4]),',',round(tb[i,5]),']。'
             )
           }
           results$descrip=descrip


           #----------plot&table---------#
           if (method == 1){
             pic=ggsurvplot(fit=fit1,data = df,
                        pval = TRUE, conf.int = (conf_int==1),
                        risk.table = (risk_table==1), # Add risk table
                        risk.table.col = "strata", # Change risk table color by group
                        ncensor.plot = (ncensor_plot==1),#Add ncensor plot
                        linetype = "strata", # Change line type by group
                        surv.median.line = "hv", # Specify median survival
                        #ggtheme = theme_bw(), # Change ggplot2 theme
                        #palette = c(rainbow(length(fit1$n))),
                        palette = style,
                        break.x.by=xlab_distance,
                        xlab=x_lab)
             #plot1表示生存曲线图
           } else{
               pic=ggsurvplot(fit=fit1,data = df,
                          conf.int = (conf_int==1),
                          risk.table = (risk_table==1),
                          risk.table.col = "strata", # Change risk table color by group
                          #ggtheme = theme_bw(), # Change ggplot2 theme
                          palette =style,
                          fun = "cumhaz",
                          ncensor.plot = (ncensor_plot==1),#Add ncensor plot
                          break.x.by=xlab_distance,
                          xlab=x_lab)
               #plot2表示累积风险图
           }

           picname <- paste0("mypictrue",mytime,"_",rand,".jpeg")
           if(risk_table==1&ncensor_plot==1){
             height=8
           }else
             if(risk_table==1|ncensor_plot==1){
               height=6
             }else{
                 height=4
               }
jpeg(file=paste0(savePath,picname),width=5,height=height,units="in",res=600)
print(pic)
dev.off()
picpath <- paste0(picname)
results$picpath <- picpath
return(results)
}
    ''')

    # 对照ref生成新的list
    temp_df = df_input[[time, statu, groups]]
    temp_df = temp_df.dropna()

    with localconverter(ro.default_converter + ro.pandas2ri.converter):
        r_df = ro.conversion.py2rpy(temp_df)
        # time=ro.FloatVector(temp_df[time])
        # statu = ro.FloatVector(temp_df[statu])
        # groups = ro.FloatVector(temp_df[groups])
    result = surv_ana(df=r_df, tim_name=time, sta_name=statu, group_name=groups, method=methods, conf_int=conf_int,
                      risk_table=risk_table,
                      ncensor_plot=ncensor_plot, xlab_distance=x_distance,
                      savePath=path, round=decimal_num, style=palette_style)

    list_plot_path = []
    with localconverter(ro.default_converter + ro.pandas2ri.converter):
        df_result = ro.conversion.rpy2py(result.rx2('table'))  # 表格
        str_result = tuple(result.rx2('descrip'))[0]  # 描述
        list_plot_path.append(tuple(result.rx2('picpath'))[0])  # 图片
    df_result.rename(columns={'X95.LCI': '95%CI下限', 'X95.UCI': '95%CI上限'}, inplace=True)
    df_result = df_result.reset_index(drop=False)
    return df_result, str_result, list_plot_path


def R_NRI_ana(df_input, pstd, pnew, gold, cut, path, decimal_num=3):
    """
    净重分类指数
    df_input：DataFrame 输入的待处理数据
    pstd： str 旧模型预测值
    pnew: str 新模型预测值
    gold: str 事件（金标准）
    cut: list 截断值
    path:str 图片保存路径
    decimal_num: int  小数点位数
    """
    NRI = ro.r('''
    NRI<-function(pstd,pnew,gold,cut,savePath,round){
      # input:  
      # pstd ：vector 旧模型预测值,
      # pnew: vector 新模型预测值
      # gold: vector 事件（金标准）
      # cut: vector 截断值
      # savePath:str 存储路径
      # round: int 小数点位数
      # return: 
      # result$picname图片名称
      # result$tab表格
      # result$describe描述
      library(survival)
      library(nricens)
      library(dplyr)
      results<-list()
      mytime <- format(Sys.time(), "%b_%d_%H_%M_%S_%Y")  #系统时间
      rand <- sample(1:100,1)
      mydata=data.frame(pstd,pnew,gold,savePath)
      if (missing(cut)) cut <- 0.5
      picname <- paste0("mypictrue",mytime,"_",rand,".jpeg")
      jpeg(file=paste0(savePath,picname),width=5,height=5,units="in",res=600)
      mode=nribin(event = gold,  p.std = pstd, p.new = pnew, updown = 'category', c=cut,
                  niter = 10000,alpha = 0.05)
      dev.off()
      results$picname<- picname
      results$tab<-round(mode$nri,round)
      # descri1<-paste0('结果如下：','NRI=',NRI,'Z=',Z,'P=',P,
      #                )
      descri2<-paste0("NRI计算结果及重抽样后估计的标准误与可信区间如表所示，NRI=",round(mode$nri[1,1],round),"。新模型较旧模型重分类正确的比例提高了",round(mode$nri[1,1],round)*100,'%。',
                        case_when(
                          mode$nri[1,1]<0 ~ "新模型预测的准确度降低了，新模型比旧模型差。",
                          mode$nri[1,1]>0 ~ "新模型预测的准确度提高了，新模型比旧模型好。",
                          mode$nri[1,1]==0 ~ "新模型、旧模型没有差异。"
                        )
                      )
    results$describe<-descri2
    return(results)
      }
    ''')
    with localconverter(ro.default_converter + ro.pandas2ri.converter):
        pstd_r = ro.FloatVector(df_input[pstd])
        pnew_r = ro.FloatVector(df_input[pnew])
        gold_r = ro.FloatVector(df_input[gold])
        cut_r = ro.FloatVector(cut)
    result = NRI(pstd=pstd_r, pnew=pnew_r, gold=gold_r, cut=cut_r, savePath=path, round=decimal_num)
    list_plot_path = []
    with localconverter(ro.default_converter + ro.pandas2ri.converter):
        df_result = ro.conversion.rpy2py(result.rx2('tab'))  # 表格
        str_result = tuple(result.rx2('describe'))[0]  # 描述
        list_plot_path.append(tuple(result.rx2('picname'))[0])  # 图片
    df_result = df_result.reset_index(drop=False)
    return df_result, str_result, list_plot_path
    
    
    
def R_RCS_ana(df_input, timevariable, statusvariable, feature_con, savePath,k=4,ti=0 ,decimal_num=3):
    """
    mydata:dataframe 需处理的数据
    timevariable:str 时间变量
    statusvariable: str 状态变量
    feature_con: str 一个定量自变量
    k:  节点数目 default=4
    ti： 设置参考点，也就是HR为1的点，常见的为中位数(default)或者临床有意义的点
    savePath:str 图片保存路径
    round:int 小数点位数

    return:
    results$rcs: strVector RCS图片路径
    results$descrip:strVector 描述结果
    """
    RCS = ro.r('''
#V1.0.5
#Author yuanke
#date 2021-7-12

#V1.0.2：第一版RCS限制性样条
RCS <- function(mydata,timevariable,statusvariable,feature_con,k=4,ti=0,savePath,round= 2
                ){
  # input:
  # mydata:dataframe 需处理的数据
  # timevariable:str 时间变量
  # statusvariable: str 状态变量
  # feature_con: str 一个定量自变量
  # k:  节点数目 default=4
  # ti： 设置参考点，也就是HR为1的点，常见的为中位数(default)或者临床有意义的点
  # savePath:str 图片保存路径
  # round:int 小数点位数
  # return:
  # results$rcs: strVector RCS图片路径
  # results$descrip:strVector 描述结果
  library(ggplot2)
  library(rms) 
  library(dplyr)
  results <- list() #结果
  mytime <- format(Sys.time(), "%b_%d_%H_%M_%S_%Y")  #时间
  rand <- sample(1:100,1)
  #开始正式画图
  mydt<-data.frame(mydata[timevariable],mydata[statusvariable],mydata[feature_con])
  colnames(mydt)<-c('time','status','yinsu')
  dd <<- datadist(mydt) #为后续程序设定数据环境
  options(datadist='dd') #为后续程序设定数据环境
  if (ti==0) ti <- median(mydt[,3])
  #拟合cox回归模型，注意这里的R命令是“cph”，而不是常见的生存分析中用到的“coxph"命令
  formula <- paste0('Surv(',timevariable,',',statusvariable,')','~','rcs(',feature_con,',',k,')')
  k<<-k
  fit<- cph(Surv(time,status)~rcs(yinsu,k),data=mydt) 
  dd$limits$yinsu[2] <- ti
  #这里是设置参考点，也就是HR为1的点，常见的为中位数或者临床有意义的点 
  fit=update(fit)
  HR <- Predict(fit,yinsu,fun=exp,ref.zero = TRUE)#只是返回exp
  P1<-ggplot(HR) #用ggplot2直接画图
  #画图
  P2<-ggplot()+geom_line(data=HR, aes(yinsu,yhat),linetype="solid",size=1,alpha = 0.7,colour="red")+
    geom_ribbon(data=HR, aes(yinsu,ymin = lower, ymax = upper),alpha = 0.1,fill="red")
  #进一步设置图形
  P2<-P2+theme_classic()+geom_hline(yintercept=1, linetype=2,size=1)+ 
    labs(title = "RCS", x=feature_con, y="HR (95%CI)") 
  rcsname <- paste0("RCS",mytime,"_",rand,".jpeg")
  jpeg(file=paste0(savePath,rcsname),width=7,height=8,units="in",res=600)
  plot(P2)
  dev.off()
  results$rcs <- rcsname
  anv<-anova(fit)#下面对非线性进行检验，调出p值
  des <- paste0('模型的P-Nonlinear为',round(anv[2,3],round),',',
                case_when(
                  anv[2,3]<0.05 ~ paste0(feature_con,'与',statusvariable,'之间存在非线性关系'),
                  anv[2,3]>=0.05 ~ paste0(feature_con,'与',statusvariable,'之间存在线性关系,建议使用线性回归')
                )
                )
  results$descrip <- des
  return(results)
}    
    ''')
    with localconverter(ro.default_converter + ro.pandas2ri.converter):
        r_mydata = ro.conversion.py2rpy(df_input)
    timevariable_r = timevariable
    statusvariable_r = statusvariable
    feature_con_r = feature_con
    result = RCS(mydata=r_mydata,timevariable=timevariable_r, statusvariable=statusvariable_r, feature_con=feature_con_r, k=k,ti=ti, savePath=savePath, round=decimal_num)
    list_plot_path = []
    with localconverter(ro.default_converter + ro.pandas2ri.converter):
        str_result = tuple(result.rx2('describe'))[0]  # 描述
        list_plot_path.append(tuple(result.rx2('rcs'))[0])  # 图片
    return str_result, list_plot_path