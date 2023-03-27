import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from collections import namedtuple
import os

def savefig(fig,fig_ttl,path=None,rp=False,facecolor='w',transparent=False):
    '''
    fig: Figure object matplotlib.figure.Figure
    fig_ttl: figure title string (some specila characterls will be removed from the file name)
    '''
    for ch in [' ','[',']','=','(',')','/']:
        if ch in fig_ttl:
            fig_ttl=fig_ttl.replace(ch,'-')
    fig_ttl=fig_ttl.replace('.','p')
    if path==None:
        filename=fig_ttl
    else:
        filename=path+fig_ttl
    if not(rp):
        if os.path.isfile(filename+'.png'):
            usr=input('Replace existing file?: ')
            if usr=='y':
                rp=True
        else:
            rp=True
    if rp:
        if transparent:
            fig.savefig(filename+'.png', format='png', dpi=200,bbox_inches='tight',transparent=transparent)
        else:
            fig.savefig(filename+'.png', format='png', dpi=200,facecolor='w',bbox_inches='tight')
        print(filename+'.png SAVED.')

def plot_BAvsScat(x_input,y_input,label='',xy_labels = None,saveplot=0, scat=1, binscale=1.0, xx_unc_modl=np.sqrt(0.5), yy_unc_modl=np.sqrt(0.5)):
    '''
    Routine to plot, side by side, 2D histograms of paired data as bland-altman and scatterplot. 
    It also calculates relevant analysis metrics.
    Takes as input:
    x_input: array of X data values
    y_input: corresponding array of Y data values, ie [X[1],Y[1]]. len(x_input)=len(y_input)
    label: text label for plotting, default is emtpy ('')
    saveplot: set to 1 to save plot, default is (0)
    scat: default (0) is to make a 2d histogram, regular scatterplot if set to 1
    binscale: default (1) is a scaling factor for how many bins to include in a 2d histogram
    xx_unc_modl and yy_unc_modl: uncertainty models for xx and yy. Used for normalization of the bias. 
        Default value corresponds to no normalization.
    xy_labels: ['ARM','MOD']
    Returns:
    scaleindependence: binary result of test for scale independence
    meanbias: mean bias (NaN if not scale independent)
    loa: one sigma Limits of agreement ([NaN,NaN] if not scale independent)
    slope: linear regression of x vs y, slope
    yintercept: linear regression of x vs y ,intercept
    lincorr: linear correlation (pearson rho)
    rankcorr: rank correlation (spearman rho)
    rmse: root mean square error
    
    '''
    #deal with inputs
    xx=np.asarray(x_input)
    yy=np.asarray(y_input)
    
    #compute Bland-Altman axes
    jj=(xx+yy)/2  #paired mean
    #kk=yy-xx      #bias
    kk= (yy-xx) / np.sqrt((xx_unc_modl**2)+(yy_unc_modl**2)) #bias, with normalization if xx_unc_modl and yy_unc_modl are set

    #compute Bland-Altman metrics
    meanbias=np.mean(kk)
    stdbias=np.std(kk)
    LOAlow= meanbias - stdbias #lower limit of agreement (LOA)
    LOAhgh= meanbias + stdbias #upper limit of agreement (LOA)

    #stuff to make plotting nice  
    nbin=int(0.5*binscale*np.sqrt(len(xx))) #find appropriate binning dimensions, relevant for 2dhist
    min_kk=meanbias - 5.*(stdbias)
    max_kk=meanbias + 5.*(stdbias) 
    
    if scat == 0:
        jj_sorted= np.sort(jj)  # sort paired mean data in ascending order
        min_jj=jj_sorted[int(0.01*len(xx))]
        max_jj=jj_sorted[int(0.99*len(xx))]
    else:
        min_jj=min(jj)
        max_jj=max(jj)
    gamma=0.5

    #check for bland-altman bias scale independence
    ba_stat, ba_p = stats.spearmanr(jj, kk)
    ba_independ = ba_p > 0.05 #check that the p-value is greater than 0.05
    if ba_independ:
        print('Bias INDEPENDENT of paired mean, r:%.3f' % ba_stat)
    else:
        print('Bias DEPENDENT on paired mean, r:%.3f' % ba_stat)
        ba_regress_result=stats.linregress(jj,kk)
        ba_min_fit_yy=ba_regress_result.slope*min_jj + ba_regress_result.intercept
        ba_max_fit_yy=ba_regress_result.slope*max_jj + ba_regress_result.intercept

    #compute scatterplot regression metrics
    regress_result=stats.linregress(xx, yy)
    spearman_r=stats.spearmanr(xx,yy)
    r_spear=round((spearman_r.correlation)*10000)/10000.
    pearson_r=stats.pearsonr(xx,yy)
    r_pear=round((pearson_r[0])*10000)/10000.
    rmse_all = np.sqrt(sum(kk**2)/len(kk)) #root mean square error
    rmse=round(rmse_all*10000)/10000.

    #set up colors
    if scat == 0:
        lineclr='white'
        loaclr='yellow'
        fitclr='cyan'
    else: 
        lineclr='black'
        loaclr='green'
        fitclr='magenta'
    
    #make nice labels for plotting
    #...for Bland-Altman
    meanbias=round(meanbias*100000)/100000.
    LOAlow=round(LOAlow*10000)/10000.
    LOAhgh=round(LOAhgh*10000)/10000.  
    ba_rankcorr=round((ba_stat)*10000)/10000.
    if ba_independ:
        txt1='Mean bias: '+str(meanbias)+' ('+fitclr+' dashed line)'
        txt2='LOA: ['+str(LOAlow)+','+str(LOAhgh)+'] ('+loaclr+' dashed lines)'
    else:
        txt1='Bias DEPENDENT on paired mean (magenta is linear fit)'
        txt2='Rank correlation: '+str(ba_rankcorr)
        
    #...regression
    slope=round(regress_result.slope*10000)/10000.
    intercept=round(regress_result.intercept*10000)/10000.
    txt3='y = '+str(slope)+'x + '+str(intercept)+' ('+fitclr+' dashed line)'
    txt4='Linear correlation: '+str(r_pear)
    txt5='Rank correlation: '+str(r_spear)
    txt6='Root Mean Square Error: '+str(rmse)
       
    #plotting setup
    
    fig, (ax1, ax2) = plt.subplots(1, 2,figsize=(16,6))
    fig.suptitle(label, fontsize=22)    

    
    #plot bland-altman
    if scat == 0:
        h=ax1.hist2d(jj,kk, bins=(nbin,nbin), norm=mcolors.PowerNorm(gamma), cmap=plt.cm.inferno,
                   range=[[min_jj, max_jj], [min_kk, max_kk]])
        fig.colorbar(h[3], ax=ax1)
    else:
        h=ax1.scatter(jj,kk)
        ax1.set_xlim([min_jj, max_jj])
        ax1.set_ylim([min_kk, max_kk])

    ax1.set_title('Bland-Altman plot', fontsize=20)
    if xx_unc_modl != np.sqrt(0.5):
        ax1.set_ylabel('Uncertainty normalized bias', fontsize=18)
    else:
        if xy_labels is None:
            ax1.set_ylabel('Bias, $y-x$', fontsize=18)
        else:
            ax1.set_ylabel("%s-%s"%(xy_labels[1],xy_labels[0]))
    if xy_labels is None:    
        ax1.set_xlabel('Paired mean, $(x+y)/2$', fontsize=18)
    else:
        ax1.set_xlabel('(%s+%s)/2'%(xy_labels[0],xy_labels[1]))

    ax1.plot([min_jj,max_jj],[0,0],color=lineclr,linestyle='solid',linewidth=4.0)
    if ba_independ:
        ax1.plot([min_jj,max_jj],[meanbias,meanbias],color=fitclr,linestyle='dashed',linewidth=3.0)
        ax1.plot([min_jj,max_jj],[LOAlow,LOAlow],color=loaclr,linestyle='dashed',linewidth=2.0)
        ax1.plot([min_jj,max_jj],[LOAhgh,LOAhgh],color=loaclr,linestyle='dashed',linewidth=2.0)
    else:
        ax1.plot([min_jj,max_jj],[ba_min_fit_yy,ba_max_fit_yy],color=fitclr,linestyle='dashed',linewidth=3.0)
    ax1.text(0.04, 0.95, txt1, horizontalalignment='left', color=lineclr,verticalalignment='center', 
             transform=ax1.transAxes,fontsize=12)
    ax1.text(0.04, 0.90, txt2, horizontalalignment='left', color=lineclr,verticalalignment='center', 
             transform=ax1.transAxes,fontsize=12)   

    #make scatterplot
    if scat == 0:
        g=ax2.hist2d(xx,yy, bins=(nbin,nbin), norm=mcolors.PowerNorm(gamma), cmap=plt.cm.inferno,
               range=[[min_jj, max_jj], [min_jj, max_jj]])
        fig.colorbar(g[3], ax=ax2)
        xax_move=0.3
    else:
        g=ax2.scatter(xx,yy)
        ax2.set_xlim([min_jj, max_jj])
        ax2.set_ylim([min_jj, max_jj])
        xax_move=0.0
        
    ax2.set_title('Scatterplot', fontsize=20)
    if xy_labels is None:
        ax2.set_xlabel('$x$', fontsize=18)
        ax2.set_ylabel('$y$', fontsize=18)
    else:
        ax2.set_xlabel(xy_labels[0])
        ax2.set_ylabel(xy_labels[1])
    ax2.plot([min_jj,max_jj],[min_jj,max_jj],color=lineclr,linestyle='solid',linewidth=4.0)

    #find regression line
    min_fit_yy=regress_result.slope*min_jj + regress_result.intercept
    max_fit_yy=regress_result.slope*max_jj + regress_result.intercept
    ax2.plot([min_jj,max_jj],[min_fit_yy,max_fit_yy],color=fitclr,linestyle='dashed',linewidth=3.0)
    ax2.text(xax_move+1.24, 0.95, txt3, horizontalalignment='left', color=lineclr,
             verticalalignment='center', transform=ax1.transAxes,fontsize=12)
    ax2.text(xax_move+1.24, 0.90, txt4, horizontalalignment='left', color=lineclr,
             verticalalignment='center', transform=ax1.transAxes,fontsize=12)
    ax2.text(xax_move+1.24, 0.85, txt5, horizontalalignment='left', color=lineclr,
             verticalalignment='center', transform=ax1.transAxes,fontsize=12)
    ax2.text(xax_move+1.24, 0.80, txt6, horizontalalignment='left', color=lineclr,
             verticalalignment='center', transform=ax1.transAxes,fontsize=12)     

    #save figure if savefig=1
    if saveplot == 1:
        figname=label+"_blandaltman"
        savefig(fig,figname)

    #violà!
    #plt.show()
    for ax in [ax1,ax2]:
        ax.tick_params(right=True,top=True)
        ax.tick_params(which='minor',length=2,right=True)
        ax.tick_params(which='minor',length=2,top=True)
        ax.tick_params(axis='both',which='both',direction='out',width=2)
        ax.tick_params(axis='both',which='major',size=5)
        ax.tick_params(axis='both',which='minor',size=3)

    #return data

    if ba_independ:
        loa=(LOAlow,LOAhgh)
    else:
        loa=(float("NAN"),float("NAN")) 
        meanbias=float("NAN")

    output = namedtuple("BlandAltman", ["scaleindependence","meanbias","loa","slope","yintercept","lincorr","rankcorr","rmse"])
    return output(
        ba_independ,
        meanbias,
        loa,
        regress_result.slope,
        regress_result.intercept,
        pearson_r[0],
        spearman_r.correlation,
        rmse_all,
    )
