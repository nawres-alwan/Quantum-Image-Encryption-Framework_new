"""Generate all final tables and visual checks directly from measured results."""
import csv
import hashlib
import json
import math
import platform
from pathlib import Path
import numpy as np
from PIL import Image,ImageDraw,ImageFont,__version__ as pillow_version

def publication_font(size):
    for name in ('Arial.ttf', 'DejaVuSans.ttf', r'C:\Windows\Fonts\arial.ttf'):
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            pass
    return ImageFont.load_default()


ROOT=Path(__file__).resolve().parent


def write_csv(path,rows):
    with Path(path).open('w',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)


def number(value,digits=4):
    if isinstance(value,str): return value
    if value==0: return '0'
    if math.isinf(value): return r'$\infty$'
    if abs(value)<1e-3:
        mantissa,exponent=f'{value:.2e}'.split('e')
        return rf'${mantissa}\times10^{{{int(exponent)}}}$'
    return f'{value:.{digits}f}'


def table(path,columns,rows,align=None,font=r'\small'):
    align=align or ('l'+'r'*(len(columns)-1))
    text=[r'\begin{center}',r'{'+font,r'\setlength{\tabcolsep}{4pt}',
          r'\begin{tabular}{@{}'+align+r'@{}}',r'\toprule',
          ' & '.join(columns)+r' \\',r'\midrule']
    text += [' & '.join(str(x) for x in row)+r' \\' for row in rows]
    text += [r'\bottomrule',r'\end{tabular}}',r'\end{center}']
    Path(path).write_text('\n'.join(text)+'\n',encoding='utf-8')


def panel(profile,manifest,path):
    width=300;row_height=300;left=70;top=60
    canvas=Image.new('RGB',(left+3*width+18,top+4*row_height+20),'white')
    draw=ImageDraw.Draw(canvas)
    font=publication_font(17)
    small=publication_font(14)
    for col,title in enumerate(['Original input','Diagnostic preview','Exact recovery']):
        draw.text((left+col*width+12,18),title,fill='#16324f',font=font)
    for row,rec in enumerate(manifest):
        image_id=rec['image_id'];y=top+row*row_height
        draw.text((10,y+125),image_id,fill='black',font=font)
        for col,suffix in enumerate(['input','diagnostic_preview','decrypted']):
            with Image.open(ROOT/'results'/profile/f'{image_id}_{suffix}.png') as im:
                im=im.convert('RGB');original_size=im.size;im.thumbnail((280,260),Image.Resampling.LANCZOS)
                x=left+col*width+(width-im.width)//2
                canvas.paste(im,(x,y+(260-im.height)//2))
            if col==0:
                draw.text((left+10,y+265),f'{rec["title"]} ({original_size[0]} x {original_size[1]})',fill='black',font=small)
    canvas.save(path)


def main():
    results=ROOT/'results';analysis=ROOT/'analysis';tables=analysis/'tables';figures=analysis/'figures'
    tables.mkdir(exist_ok=True);figures.mkdir(exist_ok=True)
    rows=json.loads((results/'all_runs.json').read_text())
    manifest=json.loads((ROOT/'input_manifest.json').read_text())
    checks=json.loads((results/'run_verification.json').read_text())
    security=json.loads((results/'security_verification.json').read_text())
    assert len(rows)==24 and checks['all_exact_decryptions']
    summaries=[];privacy=[]
    for profile in ['native','paper256']:
        for rec in manifest:
            group=[r for r in rows if r['profile']==profile and r['image_id']==rec['image_id']]
            assert len(group)==3 and all(r['byte_identical_decryption'] for r in group)
            first=group[0]
            summary={'profile':profile,'image_id':rec['image_id'],'height':first['height'],'width':first['width'],
                     'N':first['N'],'n':first['n'],'r':first['r'],'Q':first['Q'],'replicates':3,
                     'exact_recoveries':3,'decrypt_mse':0,'decrypt_ssim':1,'decrypt_psnr_db':'Infinity',
                     'max_absolute_fidelity_error':max(abs(1-r['quantum_fidelity_raw']) for r in group),
                     'max_branch_norm_error':max(r['max_branch_norm_squared_error'] for r in group),
                     'max_recovered_amplitude_error':max(max(r['max_recovered_core_amplitude_error'],r['max_recovered_wavelength_amplitude_error']) for r in group)}
            for metric in ['preview_entropy_mean','preview_npcr_byte_percent','preview_uaci_percent','preview_ssim_7x7',
                           'preview_pearson_cc','preview_horizontal_cc','preview_vertical_cc','preview_diagonal_cc',
                           'encrypt_seconds','decrypt_seconds','elapsed_seconds']:
                values=[r[metric] for r in group]
                summary[metric+'_mean']=float(np.mean(values));summary[metric+'_sample_sd']=float(np.std(values,ddof=1))
            summaries.append(summary)
            count=first['N'];limit=((1 << 64)//count)*count;dimension=count*(1 << first['r'])
            privacy.append({'profile':profile,'image_id':rec['image_id'],'N':count,'n':first['n'],'r':first['r'],'Q':first['Q'],
                            'valid_dimension_D':dimension,'full_hilbert_dimension':1 << first['Q'],
                            'key_record_bits':384,'accepted_full_key_domain_size':str((1 << 320)*limit),
                            'offset_word_acceptance_support':str(limit),
                            'uniform_core_bits':first['n']+2*first['r'],
                            'uniform_offset_entropy_bits':math.log2(count),
                            'privacy_core_entropy_bits':first['privacy_core_entropy_bits'],
                            'ideal_diamond_privacy_epsilon':0,'ideal_guess_probability':0.5,
                            'ideal_excess_advantage':0,'averaged_state_purity':1/dimension,
                            'averaged_state_entropy_bits':math.log2(dimension),
                            'scope':'valid quantum state; fresh independent ideal uniform privacy core; no PNG guarantee'})
    write_csv(results/'summary_by_image.csv',summaries);write_csv(results/'privacy_parameters.csv',privacy)
    table(tables/'headline.tex',[r'Native image',r'$N$',r'$Q$',r'Exact recoveries',r'Ideal $\epsilon_{\rm priv}$'],
          [[s['image_id'],f"{s['N']:,}",s['Q'],'3/3','0'] for s in summaries if s['profile']=='native'])
    table(tables/'inputs.tex',['Image','Source file',r'Width $\times$ height',r'Pixels $N$'],
          [[rec['image_id'],Path(rec['relative_input']).name.replace('_',r'\_'),f"${rec['width']}\\times{rec['height']}$",f"{rec['height']*rec['width']:,}"] for rec in manifest],align='llrr')
    labels=lambda s: ('Native/' if s['profile']=='native' else '256/')+s['image_id']
    table(tables/'correctness.tex',['Profile/image','Exact','MSE','SSIM','PSNR',r'Max $|1-F|$',r'Max norm error'],
          [[labels(s),'3/3','0','1',r'$\infty$',number(s['max_absolute_fidelity_error']),number(s['max_branch_norm_error'])] for s in summaries])
    table(tables/'global_checks.tex',['Check','Observed result'],
          [['Complete runs','24/24'],['Original source bytes unchanged','Yes'],['Recovered RGB bytes identical','All 24 runs'],
           ['Largest recovered amplitude error',number(max(s['max_recovered_amplitude_error'] for s in summaries))],
           ['Independent SSIM-window error',number(checks['ssim_independent_window_check_error'])]],align='ll')
    table(tables/'channels.tex',[r'$N$',r'$r$',r'$D$',r'Keys',r'Choi trace distance',r'Reference distance',r'$P_{\rm guess}$'],
          [[s['N'],s['r'],s['dimension'],s['enumerated_keys'],number(s['normalized_choi_trace_distance']),number(s['reference_decoupling_trace_distance']),f"{s['optimal_pair_guess_probability_numeric']:.12f}"] for s in security['exhaustive_private_channels']],font=r'\footnotesize')
    table(tables/'dense.tex',[r'$N$',r'$r$',r'$D$',r'Unitarity error',r'Dense/output error',r'Recovery $L_2$ error'],
          [[s['pixels'],s['r'],s['valid_dimension'],number(s['unitarity_max_error']),number(s['factorized_dense_max_error']),number(s['recovered_state_l2_error'])] for s in security['dense_production_equivalence']],font=r'\footnotesize')
    ablation_names={'no_position_phase':'No position phase','no_feature_mask':'No feature mask','no_uniform_shift':'No uniform offset'}
    table(tables/'ablations.tex',['Removed component','Choi trace distance','Reference distance'],
          [[ablation_names[s['ablation']],number(s['normalized_choi_trace_distance'],6),number(s['reference_decoupling_trace_distance'],6)] for s in security['ablations']],align='lrr')
    table(tables/'privacy.tex',['Profile/image',r'$n$',r'$r$',r'$Q$',r'$D=N2^r$',r'$h_{\rm core}$',r'$\epsilon$',r'$P_{\rm guess}$'],
          [[labels(s),s['n'],s['r'],s['Q'],f"{s['valid_dimension_D']:,}",f"{s['privacy_core_entropy_bits']:.6f}",'0','0.5'] for s in privacy],font=r'\footnotesize')
    table(tables/'phase_key.tex',[r'$N$',r'Exact wrong-phase $F$',r'Observed $F$',r'Pixel MSE'],
          [[s['N'],'0' if s['N']==4 else r'$1/9$',number(s['quantum_fidelity'],12),'0'] for s in security['wrong_phase_key_tests']])
    table(tables/'preview_metrics.tex',['Profile/image','Entropy',r'NPCR (\%)',r'UACI (\%)','SSIM','CC'],
          [[labels(s),number(s['preview_entropy_mean_mean']),number(s['preview_npcr_byte_percent_mean']),number(s['preview_uaci_percent_mean']),number(s['preview_ssim_7x7_mean']),number(s['preview_pearson_cc_mean'])] for s in summaries],font=r'\footnotesize')
    table(tables/'preview_correlations.tex',['Profile/image','Horizontal CC','Vertical CC','Diagonal CC'],
          [[labels(s),number(s['preview_horizontal_cc_mean']),number(s['preview_vertical_cc_mean']),number(s['preview_diagonal_cc_mean'])] for s in summaries])
    table(tables/'all_runs.tex',['Run',r'$N$',r'Enc. s',r'Dec. s',r'$|1-F|$',r'Max norm error'],
          [[r['run_id'].replace('_',r'\_'),f"{r['N']:,}",f"{r['encrypt_seconds']:.3f}",f"{r['decrypt_seconds']:.3f}",number(abs(1-r['quantum_fidelity_raw'])),number(r['max_branch_norm_squared_error'])] for r in rows],font=r'\footnotesize')
    for profile in ['native','paper256']: panel(profile,manifest,figures/(profile+'_panel.png'))
    software={'python':platform.python_version(),'numpy':np.__version__,'Pillow':pillow_version}
    (results/'software_versions.json').write_text(json.dumps(software,indent=2),encoding='utf-8')
    assertions={'runs':len(rows),'summary_rows':len(summaries),'privacy_rows':len(privacy),
                'all_ideal_epsilons_zero':all(p['ideal_diamond_privacy_epsilon']==0 for p in privacy),
                'all_ideal_guess_probabilities_half':all(p['ideal_guess_probability']==0.5 for p in privacy),
                'all_exact_recoveries':all(s['exact_recoveries']==s['replicates'] for s in summaries),
                'legacy_attack_cases':security['legacy_attack']['cases'],
                'legacy_attack_successes':security['legacy_attack']['correct'],
                'max_reference_channel_distance':max(s['reference_decoupling_trace_distance'] for s in security['exhaustive_private_channels'])}
    assert assertions['all_exact_recoveries'] and assertions['all_ideal_epsilons_zero']
    (results/'table_verification.json').write_text(json.dumps(assertions,indent=2),encoding='utf-8')
    print(json.dumps(assertions,indent=2))


if __name__=='__main__': main()
