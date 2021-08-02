# V 1.0.0
# date 2021-1-28
# author：xuyuan

# V1.0.0 更新说明： 初始R_multi_comp

import pandas as pd
import rpy2.robjects as ro
from rpy2.robjects.packages import importr
from rpy2.robjects import pandas2ri
from rpy2.robjects.conversion import localconverter
import AnalysisFunction.X_5_R_SmartPlot as x5r
import re
import datetime

def R_multi_comp(df_input, group, continuous_feature,path, method='LSD',decimal_num=3,style=1,palette_style='nejm'):


    # df_input: dataframe 需处理的数据
    # continuous_feature:str 分析变量
    # group:str 分组变量
    # path:str 图片保存路径
    # method: str 多重比较方法（LSD、Bonferroni、SNK、Tukey、Duncan、Scheffe）
    # decimal_num:int 小数点位数
    # style: int 森林图风格
    # palette_style: str 调色风格（nejm、lancet、jama、npg）

    multi_comp = ro.r('''
   #V1.0.3
#Author liuyue
#date 2021-2-1

#V1.0.0：第一版多重比较模型
#V1.0.1
#V1.0.2
#V1.0.3： 新增sty_col

multi_comp <- function(mydata, ana_var, gro_var,method = "LSD", sty_col = "nejm",savePath,round){
  # input:
  # mydata: dataframe 需处理的数据
  # ana_var:strVector 分析变量
  # gro_var:str 分组变量
  # method: str 多重比较方法（LSD、Bonferroni、SNK、Tukey、Duncan、Scheffe）
  # sty_col:图片风格（nejm、lancet、jama、npg）
  # savePath:str 图片保存路径
  # round:int 小数点位数
  # return:
  # results$table:dataframe 表格结果
  # results$boxplot: strVector 箱线图图片路径
  # results$descrip:strVector 描述结果
  library(agricolae)
  library(dplyr)
  library(stringr)
  library(ggplot2)
  library(ggsci)
  results <- list() #结果
  mytime <- format(Sys.time(), "%b_%d_%H_%M_%S_%Y")  #时间
  rand <- sample(1:100,1)
  ana_var0 <- mydata[,ana_var]
  gro_var0 <- mydata[,gro_var]
  
  #建立方差分析模型
  model<-aov(ana_var0~gro_var0,data=mydata)
  #选择多重比较方法
  #"LSD", "Bonferroni", "SNK", "Tukey", "Duncan", "Scheffe"
  out <- switch(method, 
                LSD        = LSD.test(model, "gro_var0", p.adj="none",group = FALSE),
                Bonferroni = LSD.test(model, "gro_var0", p.adj="bonferroni",group = FALSE),
                SNK        = SNK.test(model, "gro_var0", group = FALSE, console=FALSE),
                Tukey      = HSD.test(model, "gro_var0", group = FALSE, console=FALSE),
                Duncan     = duncan.test(model, "gro_var0", group = FALSE, console=FALSE),
                Scheffe    = scheffe.test(model, "gro_var0", group = FALSE, console=FALSE)
                )
  
  #画箱线图
  pic=ggplot(data = mydata, aes(x=gro_var0,y=ana_var0,fill=factor(gro_var0)))+
    stat_boxplot(geom = "errorbar",width=0.4)+
    geom_boxplot()+
    guides(fill=F)+
    xlab(gro_var)+
    ylab(ana_var)+
    theme(panel.background = element_rect(fill = "white",color = "black"),
          panel.grid.major.x = element_blank(),
          axis.line = element_line(),
          axis.title.x = element_text(size = 16),
          axis.title.y = element_text(size = 16),
          legend.position = "top")
  if(sty_col == "nejm"){
    pic_sty = pic+scale_fill_nejm()
  }else
    if(sty_col == "lancet"){
      pic_sty = pic+scale_fill_lancet()
    }else
      if(sty_col == "jama"){
      pic_sty = pic + scale_fill_jama()
      }else{
        pic_sty = pic + scale_fill_npg()
    }
  pic_styname <- paste0("mypictrue",mytime,"_",rand,".jpeg")
  jpeg(file=paste0(savePath,pic_styname),width=7,height=8,units="in",res=600)
  print(pic_sty)
  dev.off()
  results$picname <- pic_styname

  #输出表格&添加描述
  table <- out$comparison
  tb <- cbind(table[,1], table[,4], table[,5], table[,2])
  rownames(tb)  <- rownames(table)
  colnames(tb) <- c("均数差值","95%LCI","95%UCI","p值")
  results$table <- data.frame(round(tb,round))
  descrip <- NULL
  a <- NULL
  p_adj <- out$comparison$pvalue
  for(i in 1:nrow(table)){
    if(p_adj[i]<0.05){
      a[i] <- rownames(table)[i]
    }
  }
  a<-na.omit(a)
  descrip <- paste0('不同',gro_var,'样本对于',ana_var,'呈现出显著差异，通过'
                    ,method,'方法可以得到，有着较为明显差异的组别分别为',
                    paste0(a, collapse = "；"),'，其P值均小于0.05。')
  results$descrip <-descrip
  return(results)

}
    ''')
    df_input = df_input[[group, continuous_feature]].dropna()
    with localconverter(ro.default_converter + ro.pandas2ri.converter):
        r_df = ro.conversion.py2rpy(df_input)
    result = multi_comp (mydata=r_df, ana_var=continuous_feature, gro_var=group, method=method,
                         savePath=path,round=decimal_num,sty_col=palette_style)
    list_plot_path = []
    with localconverter(ro.default_converter + ro.pandas2ri.converter):
        df_result = ro.conversion.rpy2py(result.rx2('table'))  # 表格
    str_result = tuple(result.rx2('descrip'))[0]  # 描述
    print(result.rx2('picname'))
    list_plot_path.append(str(tuple(result.rx2('picname'))[0]))  # 图片

    df_result = df_result.reset_index(drop=False)
    df_result.columns=['变量','均数差值','95%LCI','95%UCI','p值']
    try:
        froest_plot_path = x5r.R_froest_plot(df_input=df_result, title='均数差值(95%CI)', name='变量', mean="均数差值",
                                             lowerb="95%LCI", upperb="95%UCI", savePath=path, grawid=3,
                                             tickfont=1,
                                             xlabfont=1, style=style, zero=0)
        list_plot_path.append(froest_plot_path)
    except Exception as e:
        print(e)
    return df_result, str_result, list_plot_path