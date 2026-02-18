
// Inis
function ___i(l,a,r,_a){
    const v=[];
    for(let _=0;_<l.length-1;_++){
        for(let __=0;__<l[_+1];__++){
            for(let ___=0;___<l[_];___++){
                v.push(___r(r[_],l[_],l[_+1]));
            }
            v.push(0);
        }
    }
    if(_a&&_b)return {l,a,v,alpha:_a,beta:_b}
    if(_a)return {l,a,v,alpha:_a}
    if(_b)return {l,a,v,beta:_b}
    return {l,a,v}
}

function ___r(_,i,o){
    if('basic'===_)return Math.random()*2-1;
    if('normal'===_){let u=0,v=0;while(u===0)u=Math.random();while(v===0)v=Math.random();return Math.sqrt(-2*Math.log(u))*Math.cos(2*Math.PI*v);}
    if('xavier'===_)return (Math.random()*2-1)*Math.sqrt(6/(i+o));
    if('xavier_n'===_)return Math.sqrt(2/(i+o))*(()=>{let u=0,v=0;while(u===0)u=Math.random();while(v===0)v=Math.random();return Math.sqrt(-2*Math.log(u))*Math.cos(2*Math.PI*v);})();
    if('he'===_)return (Math.random()*2-1)*Math.sqrt(6/i);
    if('he_n'===_)return Math.sqrt(2/i)*(()=>{let u=0,v=0;while(u===0)u=Math.random();while(v===0)v=Math.random();return Math.sqrt(-2*Math.log(u))*Math.cos(2*Math.PI*v);})();
    if('lecun'===_)return (Math.random()*2-1)*Math.sqrt(3/i);
    if('lecun_n'===_)return Math.sqrt(1/i)*(()=>{let u=0,v=0;while(u===0)u=Math.random();while(v===0)v=Math.random();return Math.sqrt(-2*Math.log(u))*Math.cos(2*Math.PI*v);})();
}
// Inis e;



// Aktivasi
function ___a(_,d,m){
    if('relu'===_)return d.map(__=>Math.max(__,0));
    if('relu_d'===_)return d.map(__=>__>0?1:0);
    if('linear'===_)return d;
    if('linear_d'===_)return d.map(()=>1);
    if('sigmoid'===_)return d.map(__=>1/(1+Math.exp(-__)));
    if('sigmoid_d'===_){return d.map(__=>{const s=1/(1+Math.exp(-__));return s*(1-s)})};
    if('tanh'===_)return d.map(__=>Math.tanh(__));
    if('tanh_d'===_)return d.map(__=>{const t=Math.tanh(__);return 1-(t*t)});
    if('leaky'===_)return d.map(__=>__>0?__:__*m.alpha);
    if('leaky_d'===_)return d.map(__=>__>0?1:m.alpha);
    if('elu'===_)return d.map(__=>__>0?__:m.alpha*(Math.exp(__)-1));
    if('elu_d'===_)return d.map(__=>__>0?1:m.alpha*Math.exp(__));
    if('selu'===_){const la=1.0507,al=1.67326;return d.map(__=>__>0?la*__:la*al*(Math.exp(__)-1));}
    if('selu_d'===_){const la=1.0507,al=1.67326;return d.map(__=>__>0?la:la*al*Math.exp(__));}
    if('gelu'===_)return d.map(__=>0.5*__*(1+Math.tanh(Math.sqrt(2/Math.PI)*(__+0.044715*Math.pow(__,3)))));
    if('gelu_d'===_)return d.map(__=>{const c=Math.sqrt(2/Math.PI), g=0.044715*Math.pow(__,3), t=Math.tanh(c*(__+g));return 0.5*(1+t)+0.5*__*(1-t*t)*c*(1+3*0.044715*__*__);});
    if('swish'===_)return d.map(__=>__/(1+Math.exp(-__*m.beta)));
    if('swish_d'===_)return d.map(__=>{const s=1/(1+Math.exp(-__*m.beta));return s+__*m.beta*s*(1-s);});
    if('mish'===_)return d.map(__=>__*Math.tanh(Math.log(1+Math.exp(__))));
    if('mish_d'===_)return d.map(__=>{const e=Math.exp(__), n=e*e+2*e+2, d2=e*e+2*e+1;const w=4*(e+1)+4*e*e+e*e*e+Math.pow(e,4);return (e*(w))/(Math.pow(n,2));});
    if('softplus'===_)return d.map(__=>Math.log(1+Math.exp(__)));
    if('softplus_d'===_)return d.map(__=>1/(1+Math.exp(-__)));
    if('softsign'===_)return d.map(__=>__/(1+Math.abs(__)));
    if('softsign_d'===_)return d.map(__=>1/Math.pow(1+Math.abs(__),2));
    if('hardtanh'===_)return d.map(__=>Math.max(-1,Math.min(1,__)));
    if('hardtanh_d'===_)return d.map(__=>__>-1&&__<1?1:0);
    if('softmax'===_){
        const max=Math.max(...d), exp=d.map(__=>Math.exp(__-max)), sum=exp.reduce((a,b)=>a+b,0);
        return exp.map(__=>__/sum);
    }
    if('softmax_d'===_)return d.map(()=>1);
}

// Aktivasi e;


// Prediksi
function ___p(i,m,_m){
    let a=[[...i]],d=[[]],_i=0,w=[];
    for(let _=0;_<m.l.length-1;_++){
        let i_=[];
        for(let __=0;__<m.l[_+1];__++){
            let s=0;
            for(let ___=0;___<m.l[_];___++){
                s+=i[___]*m.v[_i];_i++;w.push(1);
            }
            i_.push(s+m.v[_i]);_i++;w.push(0);
        }
        i_=___ca(m.c.clip_a,i_,m,m.a[_]);
        if(!_m){a.push(___a(m.a[_],i_,m));d.push(___a(m.a[_]+'_d',i_,m))}
        i=___a(m.a[_],i_,m);
    }
    if('end'===_m)return i;
    return {a,d,w}
}
// Prediksi e;

// Cliping a
function ___ca(_,x,m,a){
    if('none'===_)return x;
    if('value'===_)return x.map(__=>Math.max(m.c.min, Math.min(m.c.max, __)));
    if('norm'===_){
        const norm=Math.sqrt(x.reduce((a,b)=>a+b*b,0));
        if(norm>m.c.threshold) return x.map(__=>__*(m.c.threshold/norm));
        return x;
    }
    if('auto'===_){
        if(['sigmoid','tanh','exp','elu','softmax'].includes(a)) return x.map(__=>Math.max(-15, Math.min(15, __)));
        return x;
    }
}
// Cliping a e;

// Loss
function ___l(_,p,t){
    const _p=p.a[p.a.length-1], n=_p.length;
    if('mse'===_)return _p.map((__p,i)=>Math.pow(__p-t[i],2)).reduce((a,b)=>a+b,0)/n;
    if('mse_d'===_)return _p.map((__p,i)=>(__p-t[i])*2/n);
    if('mae'===_)return _p.map((__p,i)=>Math.abs(__p-t[i])).reduce((a,b)=>a+b,0)/n;
    if('mae_d'===_)return _p.map((__p,i)=>__p>t[i]?1/n:__p<t[i]?-1/n:0);
    if('bce'===_)return -_p.map((__p,i)=>t[i]*Math.log(__p+1e-15)+(1-t[i])*Math.log(1-__p+1e-15)).reduce((a,b)=>a+b,0)/n;
    if('bce_d'===_){
        p.d[p.d.length-1]=p.d[p.d.length-1].map(()=>1);
        return _p.map((__p,i)=>(__p-t[i])/n);
    }
    if('ce'===_)return -_p.map((__p,i)=>t[i]*Math.log(__p+1e-15)).reduce((a,b)=>a+b,0)/n;
    if('ce_d'===_){
        p.d[p.d.length-1]=p.d[p.d.length-1].map(()=>1);
        return _p.map((__p,i)=>(__p-t[i])/n);
    }
    if('hinge'===_)return _p.map((__p,i)=>Math.max(0,1-t[i]*__p)).reduce((a,b)=>a+b,0)/n;
    if('hinge_d'===_)return _p.map((__p,i)=>t[i]*__p<1?-t[i]/n:0);
    if('huber'===_){const d=_p.map((__p,i)=>Math.abs(__p-t[i]));return d.map(__=>__<=1?0.5*__*__:__-0.5).reduce((a,b)=>a+b,0)/n;}
    if('huber_d'===_){return _p.map((__p,i)=>{const d=__p-t[i];return (Math.abs(d)<=1?d:(d>0?1:-1))/n;});}
}
// Loss e;


// Optimizer
function ___o(_,g,m){
    if('sgd'===_)for(let __=0;__<g.length;__++)m.v[__]-=g[__]*m.c.lr;
    if('momentum'===_){
        if(!m.c.m.v)m.c.m.v=m.v.map(()=>0);
        for(let __=0;__<g.length;__++){
            m.c.m.v[__]=(m.c.m.b*m.c.m.v[__])+g[__];
            m.v[__]-=m.c.m.v[__]*m.c.lr;
        }
    }
    if('nesterov'===_){
        if(!m.c.n.v)m.c.n.v=m.v.map(()=>0);
        for(let __=0;__<g.length;__++){
            let v_prev=m.c.n.v[__];
            m.c.n.v[__]=(m.c.n.b*m.c.n.v[__])+g[__];
            m.v[__]-=((m.c.n.b*v_prev)+(1+m.c.n.b)*g[__])*m.c.lr;
        }
    }
    if('adagrad'===_){
        if(!m.c.ag.v)m.c.ag.v=m.v.map(()=>0);
        for(let __=0;__<g.length;__++){
            m.c.ag.v[__]+=g[__]*g[__];
            m.v[__]-=(m.c.lr*g[__])/(Math.sqrt(m.c.ag.v[__])+1e-8);
        }
    }
    if('rmsprop'===_){
        if(!m.c.r.v)m.c.r.v=m.v.map(()=>0);
        for(let __=0;__<g.length;__++){
            m.c.r.v[__]=m.c.r.b2*m.c.r.v[__]+(1-m.c.r.b2)*g[__]*g[__];
            m.v[__]-=(m.c.lr*g[__])/(Math.sqrt(m.c.r.v[__])+m.c.r.e);
        }
    }
    if('adadelta'===_){
        if(!m.c.ad.v)m.c.ad.v=m.v.map(()=>0);
        if(!m.c.ad.m)m.c.ad.m=m.v.map(()=>0);
         for(let __=0;__<g.length;__++){
            m.c.ad.v[__]=m.c.ad.b2*m.c.ad.v[__]+(1-m.c.ad.b2)*g[__]*g[__];
            let dx=Math.sqrt((m.c.ad.m[__]+m.c.ad.e)/(m.c.ad.v[__]+m.c.ad.e))*g[__];
            m.v[__]-=dx;
            m.c.ad.m[__]=m.c.ad.b2*m.c.ad.m[__]+(1-m.c.ad.b2)*dx*dx;
        }
    }

    if('adam'===_){
        if(!m.c.a.m)m.c.a.m=m.v.map(()=>0);
        if(!m.c.a.v)m.c.a.v=m.v.map(()=>0);
        m.c.a.t++;
        for(let __=0;__<g.length;__++){
            m.c.a.m[__]=(m.c.a.b1*m.c.a.m[__])+((1-m.c.a.b1)*g[__]);
            m.c.a.v[__]=(m.c.a.b2*m.c.a.v[__])+((1-m.c.a.b2)*g[__]*g[__]);
            let m_hat=m.c.a.m[__]/(1-Math.pow(m.c.a.b1,m.c.a.t));
            let v_hat=m.c.a.v[__]/(1-Math.pow(m.c.a.b2,m.c.a.t));
            m.v[__]-=(m.c.lr*m_hat)/(Math.sqrt(v_hat)+m.c.a.e);
        }
    }
    if('adamax'===_){
        if(!m.c.ax.m)m.c.ax.m=m.v.map(()=>0);
        if(!m.c.ax.v)m.c.ax.v=m.v.map(()=>0);
        m.c.ax.t++;
        for(let __=0;__<g.length;__++){
            m.c.ax.m[__]=(m.c.ax.b1*m.c.ax.m[__])+((1-m.c.ax.b1)*g[__]);
            m.c.ax.v[__]=Math.max(m.c.ax.b2*m.c.ax.v[__], Math.abs(g[__]));
            let m_hat=m.c.ax.m[__]/(1-Math.pow(m.c.ax.b1,m.c.ax.t));
            m.v[__]-=(m.c.lr*m_hat)/(m.c.ax.v[__]+m.c.ax.e);
        }
    }
    if('nadam'===_){
        if(!m.c.nd.m)m.c.nd.m=m.v.map(()=>0);
        if(!m.c.nd.v)m.c.nd.v=m.v.map(()=>0);
        m.c.nd.t++;
        for(let __=0;__<g.length;__++){
            m.c.nd.m[__]=(m.c.nd.b1*m.c.nd.m[__])+((1-m.c.nd.b1)*g[__]);
            m.c.nd.v[__]=(m.c.nd.b2*m.c.nd.v[__])+((1-m.c.nd.b2)*g[__]*g[__]);
            let m_hat=m.c.nd.m[__]/(1-Math.pow(m.c.nd.b1,m.c.nd.t));
            let v_hat=m.c.nd.v[__]/(1-Math.pow(m.c.nd.b2,m.c.nd.t));
            let m_bar=(m.c.nd.b1*m_hat)+((1-m.c.nd.b1)*g[__]/(1-Math.pow(m.c.nd.b1,m.c.nd.t)));
            m.v[__]-=(m.c.lr*m_bar)/(Math.sqrt(v_hat)+m.c.nd.e);
        }
    }
}
// Optimizer e;



// Gradient
function ___g(p,e,m){
    let g=[],i=m.v.length-1;
    for(let _=m.l.length-1;_>0;_--){
        const _e=p.a[_-1].map(()=>0);
        for(let __=m.l[_]-1;__>-1;__--){
            g[i]=e[__]*p.d[_][__];i--;
            for(let ___=m.l[_-1]-1;___>-1;___--){
                g[i]=p.a[_-1][___]*e[__]*p.d[_][__];
                _e[___]+=e[__]*p.d[_][__]*m.v[i];i--;
            }
        }
        e=_e;
    }
    return g;
}
// Gradient e;


// Train
function ___f(e,d,m,c){
    m.c.train=true;
    for(let _e=0;_e<e;_e++){
        if(m.c.train===false) return;
        let l=0,g=m.v.map(()=>0),b=0,w;
        for(let _d of d){
            let p=___p(_d.x,m);w=p.w;
            l+=___l(m.c.loss,p,_d.y);
            const e_=___l(m.c.loss+'_d',p,_d.y);
            const _g=___g(p,e_,m);g.forEach((_,i)=>g[i]+=_g[i]);b++;
            if(b===m.c.batch){
                g.forEach((_,i)=>g[i]/=b);
                ___o(m.c.optimizer,___r_(m.c.regular,g,m,w),m);
                g.fill(0);
                b=0;
            }
        }
        if(b>0){
            g.forEach((_,i)=>g[i]/=b);
            ___o(m.c.optimizer,___r_(m.c.regular,g,m,w),m);
        }
        if(l/d.length<m.c.stop_loss)m.c.train=false;
        if(c)c({epoch:_e,loss:l/d.length,lr:m.c.lr,c:m.c});
        if(m.c.lr_decay&&_e%m.c.epoch_lr_decay===0)m.c.lr*=m.c.lr_decay;
    }
    m.c.train=false;
}
// Train e;


// Regular
function ___r_(_,g,m,w){
    if('none'===_)return g;
    if('l1'===_)return g.map((__,i)=>__ + m.c.lambda * (m.v[i]>0?1:-1) * w[i]);
    if('l2'===_)return g.map((__,i)=>__ + m.c.lambda * m.v[i] * w[i]);
    if('l1_l2'===_)return g.map((__,i)=>__ + m.c.lambda * ((m.v[i]>0?1:-1) + m.v[i]) * w[i]);
    if('elastic'===_)return g.map((__,i)=>__ + m.c.lambda * (0.5 * (m.v[i]>0?1:-1) + 0.5 * m.v[i]) * w[i]);
}
// Regular e;



// Configurasi
function ___c(){
    return{
        lr:0.2,
        loss:'mse',
        optimizer:'sgd',
        batch:1,
        regular:'none',
        lambda:0.0001,
        clip_a:'none',
        min:-15,
        max:15,
        threshold:5,
        lr_decay :1,
        epoch_lr_decay:50,
        train:false,
        stop_loss:0.0001,
        m:{
            b:0.9,
            v:''
        },
        a:{
            b1:0.9,
            b2:0.99,
            e:1e-8,
            t:0,
            v:'',
            m:''
        },
        n:{
            b:0.9,
            v:''
        },
        ag:{
            v:''
        },
        r:{
            b2:0.99,
            e:1e-8,
            v:''
        },
        ad:{
            b2:0.9,
            e:1e-8,
            v:'',
            m:''
        },
        ax:{
            b1:0.9,
            b2:0.99,
            e:1e-8,
            t:0,
            v:'',
            m:''
        },
        nd:{
            b1:0.9,
            b2:0.99,
            e:1e-8,
            t:0,
            v:'',
            m:''
        }
    }
}
// Configurasi e;




// Class MLP
class Vignette{
    constructor(){
        this.info='mlp _nidkawai';
        this.data=[];
    }
    order(l,a,r,_a){
        this.model=___i(l,a,r,_a);
        return this;
    }
    config(c){
        this.model.c=___c();
        if(c)c(this.model.c);
        return this;
    }
    autoData(type, data){
        if('normal'===type) this.data=data;
        if('2d'===type){
            this.data=[];
            for(let i=0;i<data[0].length;i++){
                this.data.push({
                    x:data[0][i],
                    y:data[1][i]
                });
            }
        }
        if('3d'===type){
            this.data=[];
            for(let i=0;i<data.length;i++){
                this.data.push({
                    x:data[i][0],
                    y:data[i][1]
                });
            }
        }
        if('1d'===type){
            this.data=[];
            let i=data[0]+data[1];
            for(let j=2;j<data.length;j+=i){
                this.data.push({
                    x:data.slice(j,j+data[0]),
                    y:data.slice(j+data[0],j+i)
                });
            }
        }
        return this;
    }
    quest(i){
        return ___p(i,this.model,'end');
    }
    quest_all(i){
        return ___p(i,this.model);
    }
    fit(e,c){
        if(!this.data.length)throw 'no data loaded';
        ___f(e,this.data,this.model,c);
        return this;
    }
    save(){
        this.model.c='';
        return JSON.stringify(this.model,null,2);
    }
    load(json){
        this.model=JSON.parse(json);
        return this;
    }
}
// End;