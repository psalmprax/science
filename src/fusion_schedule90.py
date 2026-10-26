# SPDX-License-Identifier: Apache-2.0
# Kozykorpesh public V3/run361677445, https://www.kaggle.com/code/kozykappa/quantum-ai-lb-score-6-68108?scriptVersionId=361677445
# Source-only adaptation: explicit coefficients supplied by caller; no Toolbox or binary input.
# Original notices and credited Sumit/Rudra/Alex/Kirill/Artem lineage retained below.
"""Deferred diagonal feed-forward and live-range reduction for VQCC.
SPDX-License-Identifier: Apache-2.0
Builds on the attributed signed_digit_lab and banked_lab implementations.
"""
from fractions import Fraction
from functools import lru_cache
import numpy as np
import kickmix as km
import scheduled_arithmetic90 as H
from schedule_support90 import optimized_shift
from schedule_support90 import mobius_transform


def xor_many_toffoli(b,a,c,targets):
    targets=list(targets)
    if not targets:return
    first=targets[0]
    for t in targets[1:]:b.cx(first,t)
    b.ccx(a,c,first)
    for t in reversed(targets[1:]):b.cx(first,t)


def init_anf(b,table,address,target,borrowed=()):
    """XOR lookup with leaf products written directly into dirty output wires."""
    address,target=list(address),list(target);n=len(address)
    coef=mobius_transform(table)
    required=max(0,n-2);loan=list(borrowed)[:required]
    extra=list(b.alloc_qubits(required-len(loan)))
    scratch=loan+extra
    def emit(mask,control):
        ts=[target[j] for j in range(len(target)) if (coef[mask]>>j)&1]
        if ts:
            if control is None:b.x(km.array(ts))
            else:b.cx(control,km.array(ts))
    def walk(mask,last,control):
        emit(mask,control)
        for bit in range(last+1,n):
            child=mask|(1<<bit)
            if not mask:walk(child,bit,address[bit])
            elif bit==n-1:
                ts=[target[j] for j in range(len(target)) if (coef[child]>>j)&1]
                xor_many_toffoli(b,control,address[bit],ts)
            else:
                q=scratch[mask.bit_count()-1]
                b.init_and(control,address[bit],q);walk(child,bit,q)
                b.del_and(control,address[bit],q)
    walk(0,-1,None);b.free(km.array(extra))


def phase_from_records(b,table,address,records):
    """Correct sum_i m_i f_i(address) once, after all address-preserving work."""
    address,records=list(address),list(records);n=len(address);k=n//2
    coef=mobius_transform(table)
    left={1<<j:address[j] for j in range(k)}
    masks=[m for m in range(1,1<<k) if m.bit_count()>1]
    scratch=list(b.alloc_qubits(len(masks)+n-k-1))
    for mask,q in zip(masks,scratch):
        bit=mask.bit_length()-1;b.init_and(left[mask^(1<<bit)],address[bit],q);left[mask]=q
    carry=scratch[len(masks):];parity=b.alloc_bits(1)[0]
    def emit(high,hq):
        for low in range(1<<k):
            v=coef[low|(high<<k)]
            if not v:continue
            b.bit_store(parity,False)
            while v:
                bit=(v&-v).bit_length()-1;b.cx(records[bit],parity);v&=v-1
            if not high and not low:b.neg_if(parity);continue
            b.push_condition(parity)
            if not low:b.z(hq)
            elif not high:b.z(left[low])
            else:b.cz(left[low],hq)
            b.pop_condition()
    def walk(mask,last,q):
        emit(mask,q)
        for j in range(last+1,n-k):
            child=mask|(1<<j)
            if not mask:walk(child,j,address[k+j])
            else:
                t=carry[mask.bit_count()-1];b.init_and(q,address[k+j],t)
                walk(child,j,t);b.del_and(q,address[k+j],t)
    walk(0,-1,None)
    for mask in reversed(masks):
        bit=mask.bit_length()-1;b.del_and(left[mask^(1<<bit)],address[bit],left[mask])
    b.free(km.array(scratch));b.free(parity)


def boolean_plan(values,order=(7,4,2,3,0,1,6,5),polarity=104):
    """Positive Davio XOR/AND graph. Fixed deterministic algebra, no sampling."""
    n=len(order);indices=np.arange(1<<n)
    original=sum(((indices>>j)&1)<<order[j] for j in range(n))^polarity
    truth=int.from_bytes(np.packbits(np.asarray(values,dtype=np.uint8)[original],bitorder='little').tobytes(),'little')
    nodes=[];node_ids={}
    def xor(a,b):return(a[0]^b[0],a[1]^b[1])
    @lru_cache(None)
    def rec(t,n):
        one=(1<<(1<<n))-1
        if t==0:return (0,frozenset())
        if t==one:return (1,frozenset())
        half=(1<<(1<<(n-1)))-1;lo=t&half;hi=t>>(1<<(n-1))
        left=rec(lo,n-1);right=rec(lo^hi,n-1)
        variable=(0,frozenset((n-1,)))
        if not right[1]:return xor(left,variable) if right[0] else left
        key=(n-1,right)
        if key not in node_ids:node_ids[key]=len(order)+len(nodes);nodes.append(key)
        return xor(left,(0,frozenset((node_ids[key],))))
    result=rec(truth,n)
    return dict(order=tuple(order),polarity=polarity,nodes=nodes,result=result)


def restore_boolean(b,plan,address,target):
    address=list(address);n=len(address);order=plan['order'];pol=plan['polarity']
    flipped=[address[j] for j in range(n) if(pol>>j)&1]
    if flipped:b.x(km.array(flipped))
    wires=[address[j] for j in order]
    nodes=list(b.alloc_qubits(len(plan['nodes'])));parity=b.alloc_qubits(1)[0]
    wires+=nodes
    def form(f):
        if f[0]:b.x(parity)
        for j in sorted(f[1]):b.cx(wires[j],parity)
    for i,(variable,right) in enumerate(plan['nodes']):
        form(right);b.init_and(wires[variable],parity,nodes[i]);form(right)
    constant,terms=plan['result']
    if constant:b.x(target)
    for term in sorted(terms):b.cx(wires[term],target)
    for i in reversed(range(len(nodes))):
        variable,right=plan['nodes'][i];form(right)
        b.del_and(wires[variable],parity,nodes[i]);form(right)
    b.free(km.array(nodes));b.free(parity)
    if flipped:b.x(km.array(flipped))


def build(cfg,tables,*,fused=True,anf=True,borrow_phase=True,erase_curvature=True,
          delay_high=False,gray_checkpoint=False,late_coarse=0,curve_count=2,boolean_choices=None):
    boolean_choices=boolean_choices or {}
    assert 0<=late_coarse<=3 and 1<=curve_count<=7
    def chosen_plan(values,name,default_order=(7,4,2,3,0,1,6,5),default_pol=104):
        choice=boolean_choices.get(name,{})
        return boolean_plan(values,order=choice.get("order",default_order),polarity=choice.get("polarity",default_pol))
    extra_reload=0
    lay=H.layout(cfg,tables)
    assert (lay['w1'],lay['LB'],lay['S'],lay['whi'])==(18,21,20,8)
    split,bounds=H.prepare_derivative_split(H,cfg,tables)
    b=km.CircuitBuilder();x=b.create_quantum_register(22,name='x');flags=b.alloc_qubits(5)
    H.normalize(b,x,flags);b.x(x[-1])
    c2,residual,coarse,low,high=[b.alloc_qubits(n) for n in [cfg.w2,12,7-late_coarse,21,8]]
    full_target=km.array(list(low)+list(residual)+list(c2)+list(high)+list(coarse))
    rest_target=km.array(list(residual)+list(c2)+list(coarse))
    full_table=[0]*256;rest_table=[0]*256;high_table=[0]*256;low_table=[0]*256;c2bit_table=[0]*256
    for (e,a),(k2,_,_) in tables.items():
        key=(e,a);idx=a+128*e
        common=lay['lo'][key]|split['residual'][key]<<21|k2<<33
        full_table[idx]=common|lay['hi'][key]<<(33+cfg.w2)|split['coarse'][key]<<(41+cfg.w2)
        rest_table[idx]=split['residual'][key]|k2<<12|split['coarse'][key]<<(12+cfg.w2)
        high_table[idx]=lay['hi'][key];low_table[idx]=lay['lo'][key];c2bit_table[idx]=(k2>>6)&1
    address=km.array(list(x[14:21])+[flags[0]]);delta=x[:14]
    measured=[]
    def erase(reg,table):
        ms=b.alloc_bits(len(reg));b.hmr(reg,ms)
        measured.append((list(ms),list(table)))
        return list(ms)
    def phase(reg,low_bit):
        n_ext=10-int(borrow_phase)
        ext=b.alloc_qubits(n_ext)
        word=km.array(list(reg)+list(ext)+([x[-1]] if borrow_phase else []))
        optimized_shift(b,flags,word,len(reg),False,prune=True)
        for i,q in enumerate(word):
            exponent=low_bit+i-cfg.fy-10
            if exponent<0:b.z_pow(q,Fraction(2)**(exponent+1))
        optimized_shift(b,flags,word,len(reg),True,prune=True);b.free(ext)
    extra_borrow=[]
    def adder(a,t,cin=None,sub=False):H.folded_gadd(b,a,t,cin,sub,borrowed=(x[-1],*extra_borrow))
    def inner(inverse):
        rows=cfg.rrow_list()
        for i in reversed(rows) if inverse else rows:
            shift=cfg.sh2[i]
            if shift>=0:addend=c2[shift:];target=residual;carry=c2[shift-1] if shift else None
            else:addend=c2;target=residual[-shift:];carry=None
            b.cx(delta[i],target);adder(addend,target,carry,sub=inverse);b.cx(delta[i],target)
    def outer(derivative,inverse):
        for i in range(14) if inverse else reversed(range(14)):
            shift=cfg.sh1[i];carry=derivative[shift-1] if shift else None
            b.cx(delta[i],low);adder(derivative[shift:],low,carry,sub=inverse);b.cx(delta[i],low)
    if anf:init_anf(b,full_table,address,full_target,borrowed=(x[-1],))
    else:b.init_lookup(table=full_table,address=address,target=full_target)
    if not delay_high:
        phase(high,20)
        if fused:high_ms=erase(high,high_table)
        else:b.del_lookup(table=high_table,address=address,target=high)
        b.free(high)
    else:
        assert all(v>>7==1 for v in high_table),'Constant-high carry borrowing proof failed'
        b.x(high[-1]);extra_borrow.append(high[-1])
    inner(False)
    if delay_high:
        extra_borrow.clear();b.x(high[-1])
    erased=None
    if erase_curvature:
        assert cfg.w2==7
        checkpoint_start=7-curve_count if gray_checkpoint else 6
        if gray_checkpoint:
            for j in range(checkpoint_start,6):b.cx(c2[j+1],c2[j])
        erased=km.array(list(c2[checkpoint_start:]))
        checkpoint_table=[((tables[(i//128,i%128)][0] ^ (tables[(i//128,i%128)][0]>>1)) if gray_checkpoint else tables[(i//128,i%128)][0])>>checkpoint_start for i in range(256)]
        checkpoint_ms=erase(erased,checkpoint_table);b.free(erased)
    if delay_high:
        phase(high,20)
        if fused:high_ms=erase(high,high_table)
        else:b.del_lookup(table=high_table,address=address,target=high)
        b.free(high)
    if late_coarse:
        new=b.alloc_qubits(late_coarse)
        coarse=km.array(list(coarse)+list(new))
        cv=np.array([split['coarse'][(i//128,i%128)] for i in range(256)],dtype=np.int64)
        for j in reversed(range(7-late_coarse,7)):
            pp=chosen_plan(((cv^(cv>>1))>>j)&1,f'coarse{j}')
            restore_boolean(b,pp,address,coarse[j]);extra_reload+=len(pp['nodes'])
            if j<6:b.cx(coarse[j+1],coarse[j])
    adder(residual[11:],coarse);derivative=residual[:11]+coarse
    outer(derivative,False);phase(low[cfg.gy:],cfg.gy);outer(derivative,True)
    if fused:low_ms=erase(low,low_table);b.free(low)
    if erase_curvature:
        replacement=b.alloc_qubits(7-checkpoint_start)
        c2=km.array(list(c2[:checkpoint_start])+list(replacement))
        cv=np.array([tables[(i//128,i%128)][0] for i in range(256)],dtype=np.int64)
        gray_cv=cv^(cv>>1) if gray_checkpoint else cv
        for j in reversed(range(checkpoint_start,7)):
            pp=chosen_plan((gray_cv>>j)&1,f'curve{j}')
            restore_boolean(b,pp,address,c2[j]);extra_reload+=len(pp['nodes'])
            if gray_checkpoint and j<6:b.cx(c2[j+1],c2[j])
    adder(residual[11:],coarse,sub=True);inner(True)
    rest_target=km.array(list(residual)+list(c2)+list(coarse))
    if fused:
        rest_ms=erase(rest_target,rest_table);b.free(rest_target)
        # Fold repeated coefficient measurements into the final classical mask.
        # Gray checkpoint g5=c5 XOR c6 contributes m5*c5+(m5 XOR m6)*c6.
        if erase_curvature:
            for k,m in enumerate(checkpoint_ms):
                j=checkpoint_start+k;b.cx(m,rest_ms[12+j])
                if gray_checkpoint and j<6:b.cx(m,rest_ms[12+j+1])
        records=low_ms+rest_ms[:12+cfg.w2]+high_ms+rest_ms[12+cfg.w2:]
        assert len(records)==len(full_target)+late_coarse
        phase_from_records(b,full_table,address,records)
    else:
        # The curvature checkpoint also needs its deferred phase correction.
        if erase_curvature:raise ValueError('Curvature erasure requires fused correction')
        target=km.array(list(low)+list(rest_target));table=[low_table[j]|(rest_table[j]<<21) for j in range(256)]
        b.del_lookup(table=table,address=address,target=target);b.free(target)
    b.x(x[-1]);H.unnormalize(b,x,flags);b.free(flags)
    circuit=b.finish_circuit()
    return circuit,dict(variant='deferred_feedforward',fused=fused,anf=anf,borrow_phase=borrow_phase,
                        erase_curvature=erase_curvature,delay_high=delay_high,gray_checkpoint=gray_checkpoint,
                        boolean_reload_toffoli=extra_reload,late_coarse=late_coarse,curve_count=curve_count,
                        qubits=circuit.num_qubits,static_toffoli=int(circuit.operation_statistics().num_toffoli_like),split_bounds=bounds)
