"""Postprocess frozen CSV trials; never change decisions or simulation values."""
from pathlib import Path
import csv,json,hashlib
import numpy as np
from robust_experiments import summarize,holm
from final_revision_experiments import save_csv,paired_rows
from revision_experiments import ci99

def read(path):
    with Path(path).open() as f:
        data=[]
        for row in csv.DictReader(f):
            for k,v in row.items():
                if v in ('True','False'):row[k]=v=='True';continue
                try:row[k]=float(v)
                except ValueError:pass
                if k in ('admitted','seed','failures','samples','confirmed_excess','anchors','horizon') and isinstance(row[k],float) and np.isfinite(row[k]):row[k]=int(row[k])
            data.append(row)
        return data

def main(output=Path('results')):
    for suite in ('legacy','price','markov','causal','risk','density','stress','bias'):
        parts=sorted(output.glob(f'v5_{suite}_records_part*.csv'))
        if parts:
            records=[r for path in parts for r in read(path)]
        else:
            path=output/f'v5_{suite}_records.csv'
            if not path.exists():continue
            records=read(path)
        seen=set()
        for row in records:
            key=(row['seed'],row.get('case',''),row['method'],row.get('state',''))
            if key in seen:raise ValueError('Overlapping parts: '+str(key))
            seen.add(key)
        if suite=='legacy':
            accepted=[r for r in records if r['admitted']]
            for row,p in zip(accepted,holm([r['excess_p'] for r in accepted])):
                row['holm_excess_p']=float(p);row['confirmed_excess']=int(p<=.01 and row['violation']>.1)
        save_csv(output/f'v5_{suite}_records.csv',records)
        save_csv(output/f'v5_{suite}_summary.csv',summarize(records))
        if suite=='price':
            pairs=[('GRE','Certified SPEB'),('Causal GRE','Causal certified SPEB'),
                   ('Certified SPEB','SPEB'),('Causal GRE','GRE'),('Causal GRE','SPEB')]
            paired=paired_rows(records,pairs);lookup={(r['method'],r['seed']):r for r in records}
            for row in paired:
                differences=[lookup[row['first'],s]['pilot_energy_mj']-lookup[row['second'],s]['pilot_energy_mj']
                    for s in {r['seed'] for r in records} if lookup[row['first'],s]['admitted'] and lookup[row['second'],s]['admitted']]
                row['pilot_energy_mj_difference'],row['pilot_energy_mj_ci99']=ci99(differences)
            save_csv(output/'v5_price_paired.csv',paired)
        print('assembled',suite,len(records),flush=True)
    source=Path(__file__)
    (output/'v5_assembly_metadata.json').write_text(json.dumps(dict(source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
        inference='99% layout intervals; exact binomial mission-risk intervals; Holm .01 across original SPEB layouts',
        normalization='Integer counters preserved on CSV read; no raw numerical values or decisions edited'),indent=2)+'\n')

if __name__=='__main__':main()
