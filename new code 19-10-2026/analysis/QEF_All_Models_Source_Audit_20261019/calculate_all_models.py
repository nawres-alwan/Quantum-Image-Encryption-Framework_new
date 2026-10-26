from pathlib import Path
from decimal import Decimal as D, localcontext, ROUND_CEILING, ROUND_FLOOR
import csv,json
ROOT=Path(__file__).resolve().parent
IMAGES=[('a','Ashurbanipal',256,256),('b','Hammurabi code',222,227),('c','Lamassu',256,256),('d','Babylon Lion',768,1024)]
MODELS=[('FRQI',1),('NEQR',8),('QIRHSI',10),('EQIRHSI',10),('AFQIRHSI',7),('QIRBP',6),('QRMW',10),('QIRWC',None)]
def pi_decimal():
    a,b,t,p=D(1),D(1)/D(2).sqrt(),D(1)/4,D(1)
    for _ in range(11):
        na=(a+b)/2;b=(a*b).sqrt();t-=p*(a-na)**2;a,p=na,p*2
    return (a+b)**2/(4*t)
def cost(N,r,L,precision):
    with localcontext() as ctx:
        ctx.prec=precision
        pi=pi_decimal();one=D(1);f=D(2)**r;n=D(N);c=one-one/n
        if L is None:a=2/pi
        else:
            z=pi/(2*L);sine=term=z
            for j in range(1,500):
                term*=-z*z/D((2*j)*(2*j+1));ns=sine+term
                if ns==sine:break
                sine=ns
            a=one/(D(L)*sine)
        beta=a**(6*r);dim=n*f;x=beta*f;v=x*(n-1)
        C=((1+v)*(1+v).ln()+(n-1)*(1-x)*(1-x).ln())/dim
        Cp=(1+dim*beta*beta*c).ln()
        p=D(2)/3;h=-p*p.ln()-(1-p)*(1-p).ln();J=D(2).ln()-h
        trace=(p-D('.5'))/(beta*c);information=J/C
        q=int(max(trace,information).to_integral_value(rounding=ROUND_CEILING))
        assert 0<C<=Cp*(1+D('1e-60')) and x<1
        return dict(q=q,beta=str(+beta),C_exact=str(+C),C_purity=str(+Cp),trace_threshold=str(+trace),information_threshold=str(+information))
def show(q):
    if q<1000000:return f'{q:,}'.replace(',',r'{,}')
    v=D(q);e=v.adjusted()
    with localcontext() as ctx:
        ctx.prec=30;m=(v/(D(10)**e)).quantize(D('.00001'),rounding=ROUND_FLOOR)
    return rf'{m}\times10^{{{e}}}'
def main():
    rows=[];panels=[]
    for panel,name,H,W in IMAGES:
        N=H*W;n=(N-1).bit_length();lines=[]
        for model,r0 in MODELS:
            r=r0 if r0 is not None else n+8;vals={}
            for L in [2,4,256,None]:
                val=cost(N,r,L,120);repeated=cost(N,r,L,170)
                assert val['q']==repeated['q']
                vals[L]=val
                rows.append(dict(image=name,H=H,W=W,N=N,model=model,n=n,r=r,Q=n+r,padded_dimension=2**(n+r),valid_dimension=N*2**r,angle_law='continuous_ideal' if L is None else f'finite_L{L}',required_reference_pair_entropy_bits=None if L is None else str(6*r*(D(L).ln()/D(2).ln())),actual_original_source_certified=False,**val))
            rt=str(r) if model!='QIRWC' else rf'{n}+8'
            number=f'{N:,}'.replace(',',r'{,}')
            dim=number+rf'\,2^{{{r}}}'
            lines.append(rf'\textbf{{{model}}} & \({rt}\) & \({n+r}\) & \({dim}\) & $\geq {show(vals[2]["q"])}$ & $\geq {show(vals[None]["q"])}$ & \({6*r}\) & NC \\')
        heading=rf'\textbf{{({panel}) {name} image, $H\times W={H}\times{W}$, $n=\lceil\log_2(HW)\rceil={n}$}}\\[3pt]'
        tab=r'''{\footnotesize\setlength{\tabcolsep}{4pt}
\begin{tabular}{lccccccc}
\hline\rowcolor{Bisque3}
\textbf{Model} & $\boldsymbol{r}$ & $\boldsymbol{Q=n+r}$ & $\boldsymbol{D=N2^r}$ & $\boldsymbol{q_2}$ & $\boldsymbol{q_\infty}$ & $\boldsymbol{H_2\geq}$ & \textbf{Source}\\
\hline
'''+ '\n'.join(lines)+r'''
\hline\end{tabular}}\par'''
        panels.append(heading+'\n'+tab)
    text=r'''\begin{sidewaystable}[p]
\centering\color{red}
\caption{{\color{red}Conditional QEF image-discrimination sample bounds for all listed QIR models and the four original images. The original generator is not certified by these reference calculations.}}
\label{tab:qef_quantum_attack_resistance}
\vspace{0.25cm}
'''+ '\n\n\\par\\vspace{0.3cm}\n'.join(panels)+r'''
\vspace{0.2cm}
\parbox{\textwidth}{\footnotesize\color{red}
$q_2$ and $q_\infty$ are necessary independent ciphertext preparations for success $2/3$, from Eq.~\eqref{eq:qef_final_sample_bound}; the latter is an ideal continuous reference, not a finite-key claim.
$H_2$ is the necessary conditional source entropy (bits) for the $L=2$ independent-tuple reference, not a security exponent. NC means the original source conditions are not certified.
The declared benchmark uses 8-bit intensity, AFQIRHSI $p=4$, RGB QRMW, and one wavelength label ($b=0$) for QIRBP/QIRWC. In QIRWC the second position register is included among the features, so $r=n+8$.
These are explicit mathematical settings, not reconstructed experiment metadata.
Valid dimension $D$ differs from padded $d=2^Q$ when $N<2^n$.
Large lower bounds are rounded down; exact integer ceilings are supplied separately.}
\end{sidewaystable}
'''
    (ROOT/'Table7_All_Models.tex').write_text(text,encoding='utf-8')
    with (ROOT/'Table7_All_Models.csv').open('w',encoding='utf-8-sig',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    old=[]
    for _,name,H,W in IMAGES:old.append(dict(image=name,r=11,implied_wavelength_label_qubits=5,**cost(H*W,11,None,170)))
    (ROOT/'previous_r11_recalculation.json').write_text(json.dumps(old,indent=2),encoding='utf-8')
    print(json.dumps(dict(rows=len(rows),display_rows=32,precisions=[120,170],sample_IM1=[dict(model=v['model'],r=v['r'],law=v['angle_law'],q=v['q']) for v in rows if v['image']=='Ashurbanipal' and v['angle_law'] in ['finite_L2','continuous_ideal']]),indent=2))
if __name__=='__main__':main()

