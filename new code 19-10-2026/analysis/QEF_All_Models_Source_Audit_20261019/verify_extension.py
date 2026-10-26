"""Independent verification of the strengthened QEF bounds."""
import csv,itertools,json,math
from decimal import Decimal as D,localcontext,ROUND_CEILING
from pathlib import Path
import numpy as np
import sys
sys.dont_write_bytecode=True
import verify_moment_bound as h
ROOT=Path(__file__).resolve().parent
RNG=np.random.default_rng(2026101907)

def image(N,r):
    v=RNG.normal(size=(N,1<<r))+1j*RNG.normal(size=(N,1<<r))
    v/=np.linalg.norm(v,axis=1)[:,None];v=v.flatten()/math.sqrt(N)
    return np.outer(v,v.conj())

def budget(N,r,beta):
    f=1<<r;xp=1+f*beta*(N-1);xm=1-f*beta
    return (xp*math.log(xp)+(N-1)*xm*math.log(xm))/(N*f)

def jointcheck(states,C,eps,N,tag):
    rows=[];dim=len(states[0])
    for q in range(1,4 if dim<=4 else 3):
        pair=[h.kron([s]*q) for s in states]
        pg=.5+h.tdist(*pair)/2
        chi=h.entropy((pair[0]+pair[1])/2)-(h.entropy(pair[0])+h.entropy(pair[1]))/2
        req=math.log(2)-h.binary_entropy(pg)
        assert req<=chi+1e-10 and chi<=q*C+1e-10 and pg<=.5+q*eps+1e-10
        rows.append(dict(N=N,model=tag,q=q,optimal_success=pg,Holevo=chi,
                         Fano_required=req,information_upper=q*C,
                         trace_success_upper=min(1.,.5+q*eps)))
    return rows

def references():
    rows=[];joint=[];perturb=[]
    for N,r,L in [(2,1,2),(3,1,2),(2,1,4),(2,2,2)]:
        f=1<<r;dim=N*f;angles=[2*math.pi*j/L for j in range(L)]
        lookup=[]
        for a in itertools.product(range(L),repeat=3*r):
            lookup.append(h.kron([h.rotation(h.Z,angles[a[3*j+2]])@
                h.rotation(h.Y,angles[a[3*j+1]])@h.rotation(h.X,angles[a[3*j]]) for j in range(r)]))
        mean=sum(lookup)/len(lookup);c=1/L;s=1/(L*math.tan(math.pi/(2*L)))
        single=(c*h.I-1j*s*h.Z)@(c*h.I-1j*s*h.Y)@(c*h.I-1j*s*h.X)
        me=float(np.max(np.abs(mean-h.kron([single]*r))));te=0.
        for a,b in itertools.product(range(f),repeat=2):
            x=np.zeros((f,f),complex);x[a,b]=1
            avg=sum(t@x@t.conj().T for t in lookup)/len(lookup)
            te=max(te,float(np.max(np.abs(avg-np.eye(f)*(a==b)/f))))
        W=h.public_layers(N,r,RNG);count=len(lookup)**N
        superop=np.zeros((dim*dim,dim*dim),complex)
        for inds in itertools.product(range(len(lookup)),repeat=N):
            S=np.zeros((dim,dim),complex)
            for k,ix in enumerate(inds):S[k*f:(k+1)*f,k*f:(k+1)*f]=lookup[ix]
            U=W@S;superop+=np.kron(U.conj(),U)/count
        expected=np.zeros_like(superop)
        for j in range(dim*dim):
            x=np.zeros(dim*dim,complex);x[j]=1
            out=W@h.averaged_substitution(x.reshape((dim,dim),order="F"),N,r,mean)@W.conj().T
            expected[:,j]=out.flatten(order="F")
        se=float(np.max(np.abs(superop-expected)))
        beta=(1/(L*math.sin(math.pi/(2*L))))**(6*r)
        C=budget(N,r,beta);Cp=math.log1p(dim*beta**2*(1-1/N));tau=np.eye(dim)/dim
        target=np.sort(np.array([(1+f*beta*(N-1))/dim]+[(1-f*beta)/dim]*(N-1)+[1/dim]*(N*(f-1))))
        row=dict(N=N,r=r,L=L,enumerated_keys=count,first_moment_error=me,twirl_error=te,
                 superoperator_error=se,C_exact=C,C_purity=Cp,spectrum_error=0.,entropy_error=0.)
        assert C<=Cp+1e-12 and max(me,te,se)<2e-11
        for trial in range(12):
            rho=image(N,r)
            sigma=(superop@rho.flatten(order="F")).reshape((dim,dim),order="F")
            ev=np.linalg.eigvalsh((sigma+sigma.conj().T)/2)
            rel=sum(float(v)*math.log(dim*float(v)) for v in ev if v>0)
            row["spectrum_error"]=max(row["spectrum_error"],float(np.max(np.abs(ev-target))))
            row["entropy_error"]=max(row["entropy_error"],abs(rel-C))
            assert abs(h.tdist(sigma,tau)-beta*(1-1/N))<2e-11
            assert abs(np.linalg.norm(sigma-tau,"fro")-beta*math.sqrt(1-1/N))<2e-11
            if trial==0:
                for kap in [.001,.05]:
                    altered=(1-kap)*sigma+kap*(W@rho@W.conj().T)
                    Ck=min(math.log(dim),math.log1p(dim*(beta*math.sqrt(1-1/N)+2*kap)**2))
                    actual=math.log(dim)-h.entropy(altered)
                    assert actual<=Ck+1e-11
                    assert h.tdist(altered,tau)<=beta*(1-1/N)+kap+1e-11
                    perturb.append(dict(N=N,r=r,L=L,kappa=kap,relative_entropy=actual,C_upper=Ck))
        assert max(row["spectrum_error"],row["entropy_error"])<2e-11
        rows.append(row);states=[]
        for val in [0,1]:
            v=np.zeros(dim,complex);v[np.arange(N)*f+val]=1/math.sqrt(N)
            states.append((superop@np.outer(v,v).flatten(order="F")).reshape((dim,dim),order="F"))
        joint+=jointcheck(states,C,beta*(1-1/N),N,f"reference r{r} L{L}")
    return rows,joint,perturb

def correlations():
    rows=[];joint=[]
    for N in [2,3]:
        dim=2*N;tau=np.eye(dim)/dim
        W=[h.public_layers(N,1,RNG) for _ in [0,1]];R={}
        for a in range(8):
            R[a]=[h.rotation(h.Z,2*math.pi*((a+2*k)%8)/8)@
                  h.rotation(h.Y,math.pi*((a+k)%4)/4)@
                  h.rotation(h.X,math.pi*((3*a+k)%5)/5) for k in range(N)]
        delta=eta=0.
        for b in [0,1]:
            keys=[a for a in range(8) if a%2==b]
            for k in range(N):
                bloch=np.array([[np.trace(P@sum(R[a][k]@Q@R[a][k].conj().T for a in keys)/4).real/2
                          for Q in [h.X,h.Y,h.Z]] for P in [h.X,h.Y,h.Z]])
                delta=max(delta,float(np.linalg.norm(bloch,2))/math.sqrt(2))
                for l in range(N):
                    if l!=k:
                        M=sum(np.kron(R[a][l].conj(),R[a][k]) for a in keys)/4
                        eta=max(eta,float(np.linalg.norm(M,2)))
        bound=delta**2/N+(1-1/N)*eta**2;C=math.log1p(dim*bound)
        eps=min(1.,.5*math.sqrt(dim*bound));be=0.;maxhs=0.;maxrel=0.
        def conditional(rho,b):
            out=np.zeros_like(rho)
            for a in range(8):
                if a%2!=b:continue
                S=np.zeros((dim,dim),complex)
                for k in range(N):S[2*k:2*k+2,2*k:2*k+2]=R[a][k]
                out+=S@rho@S.conj().T/4
            return out
        def average(rho):
            return sum(W[b]@conditional(rho,b)@W[b].conj().T for b in [0,1])/2
        for trial in range(20):
            rho=image(N,1)
            for b in [0,1]:
                err=conditional(rho,b)-tau
                blocks=sum(float(np.linalg.norm(err[2*k:2*k+2,2*l:2*l+2],"fro"))**2
                           for k,l in itertools.product(range(N),repeat=2))
                be=max(be,abs(float(np.linalg.norm(err,"fro"))**2-blocks))
                assert blocks<=bound+1e-12
            sigma=average(rho);hs=float(np.linalg.norm(sigma-tau,"fro"))**2
            rel=math.log(dim)-h.entropy(sigma)
            assert hs<=bound+1e-12 and rel<=C+1e-12 and h.tdist(sigma,tau)<=eps+1e-12
            maxhs=max(maxhs,hs);maxrel=max(maxrel,rel)
        rows.append(dict(N=N,r=1,records=8,conditioning="B=A mod2",image_trials=20,
                         delta2=delta,eta2_certified_upper=eta,HS_squared_upper=bound,
                         max_HS_squared=maxhs,max_relative_entropy=maxrel,C_upper=C,
                         block_orthogonality_error=be))
        states=[]
        for val in [0,1]:
            v=np.zeros(dim,complex);v[np.arange(N)*2+val]=1/math.sqrt(N)
            states.append(average(np.outer(v,v)))
        joint+=jointcheck(states,C,eps,N,"correlated actual-source model")
    return rows,joint

def pi_chudnovsky(precision):
    total=D(0)
    for k in range(precision//14+3):
        total+=D((-1)**k*math.factorial(6*k)*(13591409+545140134*k))/D(
            math.factorial(3*k)*math.factorial(k)**3*640320**(3*k))
    return D(426880)*D(10005).sqrt()/total

def g(x):
    if abs(x)>D(".1"):return (1+x)*(1+x).ln()-x
    total=D(0);power=x*x
    for m in range(2,2000):
        term=power/D(m*(m-1))*(1 if m%2==0 else -1)
        nxt=total+term
        if nxt==total:break
        total=nxt;power*=x
    return total

def independent_cost(N,r,L):
    with localcontext() as ctx:
        ctx.prec=145;pi=pi_chudnovsky(145)
        if L is None:a=2/pi
        else:
            z=pi/D(2*L);term=sine=z
            for j in range(1,300):
                term*=-z*z/D(2*j*(2*j+1));nxt=sine+term
                if nxt==sine:break
                sine=nxt
            a=1/(D(L)*sine)
        beta=a**(6*r);f=D(2)**r;n=D(N);c=1-1/n;x=f*beta
        C=(g(x*(n-1))+(n-1)*g(-x))/(n*f)
        p=D(2)/3;J=D(2).ln()+p*p.ln()+(1-p)*(1-p).ln()
        tr=(p-D(".5"))/(beta*c);info=J/C
        return int(max(tr,info).to_integral_value(rounding=ROUND_CEILING)),beta,C

def tables():
    rows=list(csv.DictReader((ROOT/"Table7_All_Models.csv").open(encoding="utf-8-sig",newline="")))
    assert len(rows)==128
    rs={"FRQI":1,"NEQR":8,"QIRHSI":10,"EQIRHSI":10,"AFQIRHSI":7,"QIRBP":6,"QRMW":10}
    maxrel=D(0)
    for row in rows:
        N=int(row["H"])*int(row["W"]);n=(N-1).bit_length()
        r=n+8 if row["model"]=="QIRWC" else rs[row["model"]]
        assert all(int(row[k])==v for k,v in dict(N=N,n=n,r=r,Q=n+r,
            valid_dimension=N*(1<<r),padded_dimension=1<<(n+r)).items())
        L=None if row["angle_law"]=="continuous_ideal" else int(row["angle_law"].replace("finite_L",""))
        q,beta,C=independent_cost(N,r,L)
        assert q==int(row["q"])
        with localcontext() as ctx:
            ctx.prec=145;rel=abs(D(row["C_exact"])-C)/C
            maxrel=max(maxrel,rel)
            assert rel<D("1e-50") and abs(D(row["beta"])-beta)<D("1e-100")
        assert row["actual_original_source_certified"]=="False"
        if L is not None:assert abs(D(row["required_reference_pair_entropy_bits"])-D(6*r*int(math.log2(L))))<D("1e-20")
    return dict(rows=128,precision=145,pi_algorithm="independent Chudnovsky",
                entropy_algorithm="positive g-series with linear terms cancelled analytically",
                maximum_relative_C_difference=str(maxrel),all_ceilings_dimensions_match=True)

def countermodels():
    fixed=[]
    for N in [65536,50394,65536,786432]:
        q=math.ceil(math.log1p(-math.pi**2/12)/math.log1p(-1/N))
        p=lambda n:.5+2/math.pi**2*(-math.expm1(n*math.log1p(-1/N)))
        assert p(q-1)<2/3<=p(q)
        fixed.append(dict(N=N,q=q,success=p(q),previous_success=p(q-1)))
    nodes,weights=np.polynomial.legendre.leggauss(48);angles=(nodes+1)*math.pi/2
    channels=[sum(w/2*np.kron(h.rotation(axis,t).conj(),h.rotation(axis,t))
        for t,w in zip(angles,weights)) for axis in [h.X,h.Y,h.Z]]
    T=channels[2]@channels[1]@channels[0];rho=np.diag([1.,0.]).astype(complex)
    out=(T@rho.flatten(order="F")).reshape((2,2),order="F")
    signal=float(np.trace(h.X@out).real);assert abs(signal-(2/math.pi)**2)<1e-12
    zrows=[]
    for N in [2,3,5]:
        dim=2*N
        for bit in [0,1]:
            v=np.zeros(dim,complex);v[np.arange(N)*2+bit]=1/math.sqrt(N);probs=[]
            for a in range(8):
                perm=RNG.permutation(N);P=np.zeros((dim,dim),complex)
                S=np.zeros_like(P);E=np.zeros_like(P)
                for k in range(N):
                    P[2*perm[k]:2*perm[k]+2,2*k:2*k+2]=h.I
                    S[2*k:2*k+2,2*k:2*k+2]=h.rotation(h.Z,(a+1)*(k+1)*math.pi/(9*N))
                    E[2*k:2*k+2,2*k:2*k+2]=h.Z if bin(k).count("1")%2 else h.I
                result=E@P@S@v
                prob=float(sum(abs(result[2*k+bit])**2 for k in range(N)))
                assert abs(prob-1)<1e-12;probs.append(prob)
            zrows.append(dict(N=N,feature_bit=bit,fresh_records=8,minimum_correct_Z_probability=min(probs)))
    return dict(pi_range_signal=signal,signal_error=abs(signal-(2/math.pi)**2),
                fixed_budget_thresholds=fixed,fresh_Z_CZ_records=zrows)

def main():
    spec,joint,pert=references();corr,cjoint=correlations()
    result=dict(reference_models=spec,reference_joint_checks=joint,source_perturbations=pert,
        correlated_source_models=corr,correlated_joint_checks=cjoint,table_audit=tables(),
        countermodel_audit=countermodels(),original_source_certified=False,
        hardware_or_cryptanalytic_attacks_executed=False,original_manuscript_modified=False)
    (ROOT/"extension_verification.json").write_text(json.dumps(result,indent=2),encoding="utf-8")
    se=max(x["superoperator_error"] for x in spec);sp=max(x["spectrum_error"] for x in spec)
    ee=max(x["entropy_error"] for x in spec);be=max(x["block_orthogonality_error"] for x in corr)
    text=(
        "\\paragraph{Independent verification of the strengthened bound.}\n"
        "Four finite reference models were checked by exhaustive substitution-key enumeration "
        "(64, 512, 4096, and 4096 records), including complete superoperators, first moments, "
        "and feature twirls. Exact spectra and relative entropies were checked on 48 normalized "
        "image states. Maximum superoperator, spectrum, and entropy errors were "
        +f"{se:.3g}, {sp:.3g}, and {ee:.3g}, respectively. "
        +f"Optimal joint binary measurements in {len(joint)} reference cases satisfied both converses. "
        "Two eight-record sources with correlated substitution and control material were checked "
        +f"on 40 image states (block-orthogonality error {be:.3g}), "
        +f"with {len(cjoint)} additional joint-measurement checks. "
        "Eight source-perturbation cases checked the robustness inequalities. "
        "All 128 table rows, dimensions, and integer ceilings were independently recomputed "
        "at 145 decimal digits using a different pi algorithm and a cancellation-resistant "
        "entropy formula. The fixed-budget range attack and 48 fresh-record Z/CZ outcomes "
        "were also checked. These calculations support the derivation; they do not certify "
        "the original source law or report cryptanalytic attacks or hardware performance.\n")
    (ROOT/"verification_summary.tex").write_text(text,encoding="utf-8")
    print(json.dumps(dict(reference_models=4,image_checks=48,reference_joint_checks=len(joint),
        correlated_images=40,correlated_joint_checks=len(cjoint),table_rows=128,
        max_superoperator_error=se,max_spectrum_error=sp,max_entropy_error=ee,max_block_error=be,
        max_table_C_difference=result["table_audit"]["maximum_relative_C_difference"])))
if __name__=="__main__":main()

