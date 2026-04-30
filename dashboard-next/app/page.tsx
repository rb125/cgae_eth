"use client";

import { useEffect, useState } from "react";
import {
  AreaChart, Area, BarChart, Bar, XAxis, YAxis, Tooltip,
  ResponsiveContainer, CartesianGrid, Cell, PieChart, Pie,
} from "recharts";

const API = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000";
const POLL_MS = 2000;

const VIOLET = "#7c3aed";
const BLUE = "#2563eb";
const GREEN = "#059669";
const RED = "#dc2626";
const AMBER = "#d97706";
const TC: Record<number,string> = {0:"#94a3b8",1:"#6366f1",2:"#2563eb",3:"#7c3aed",4:"#d97706",5:"#dc2626"};

interface Economy { aggregate_safety:number; active_agents:number; total_balance:number; total_earned:number; contracts_completed:number; contracts_failed:number }
interface Agent { agent_id:string; model_name:string; strategy:string; current_tier:number; balance:number; total_earned:number; total_penalties:number; contracts_completed:number; contracts_failed:number; status:string; wallet_address?:string; ens_name?:string; robustness:{cc:number;er:number;as_:number;ih:number}|null }
interface Trade { round:number; agent:string; task_id:string; task_prompt:string; tier:string; domain:string; passed:boolean; reward:number; penalty:number; token_cost:number; latency_ms:number; output_preview:string; constraints_passed:string[]; constraints_failed:string[] }
interface Evt { timestamp:number; type:string; agent:string; message:string }

function usePoll<T>(url:string,ms:number):T|null{const[d,setD]=useState<T|null>(null);useEffect(()=>{let a=true;const p=()=>{fetch(url).then(r=>r.json()).then(v=>{if(a)setD(v)}).catch(()=>{})};p();const id=setInterval(p,ms);return()=>{a=false;clearInterval(id)}},[url,ms]);return d}

/* ---- Atoms ---- */
const Card=({children,className=""}:{children:React.ReactNode;className?:string})=>(
  <div className={`bg-white/80 backdrop-blur-sm border border-white/60 rounded-2xl shadow-lg shadow-black/[0.03] ${className}`}>{children}</div>);

const Stat=({label,value,sub,pulse}:{label:string;value:string;sub?:string;pulse?:boolean})=>(
  <Card className="p-5">
    <div className="flex items-center justify-between mb-1"><p className="text-[11px] uppercase tracking-wider text-slate-400 font-semibold">{label}</p>{pulse&&<span className="h-2 w-2 rounded-full bg-emerald-500 animate-pulse"/>}</div>
    <p className="text-2xl font-extrabold text-slate-800">{value}</p>{sub&&<p className="text-[11px] text-slate-400 mt-0.5">{sub}</p>}
  </Card>);

const TB=({t}:{t:number})=>{const c=TC[t]||"#94a3b8";return<span className="px-2 py-0.5 rounded-full text-[10px] font-bold" style={{background:c+"18",color:c,border:`1px solid ${c}35`}}>T{t}</span>};
const RB=({l,v}:{l:string;v:number})=>{const p=Math.round(v*100);const c=v>=.65?GREEN:v>=.4?AMBER:RED;return(<div className="flex items-center gap-1.5"><span className="w-5 text-[10px] text-slate-400 font-medium">{l}</span><div className="flex-1 h-1.5 bg-slate-100 rounded-full overflow-hidden"><div className="h-full rounded-full transition-all duration-500" style={{width:`${p}%`,backgroundColor:c}}/></div><span className="w-7 text-right text-[10px] text-slate-500 font-medium">{p}%</span></div>)};
const Addr=({id}:{id:string})=>(!id||id.length<8?<span className="text-slate-400 font-mono text-[10px]">{id}</span>:<span className="text-slate-400 font-mono text-[10px]">{id.slice(0,6)}…{id.slice(-4)}</span>);
const tt={backgroundColor:"#fff",border:"1px solid #e2e8f0",borderRadius:12,fontSize:11,color:"#1e293b",boxShadow:"0 4px 12px rgba(0,0,0,0.06)"};
const StatusDot=({s}:{s:string})=>{const c:Record<string,string>={idle:"#94a3b8",setup:AMBER,running:GREEN,done:BLUE,connecting:"#94a3b8"};return<span className="inline-flex items-center gap-1.5 text-xs text-slate-500 font-medium"><span className="h-2 w-2 rounded-full" style={{backgroundColor:c[s]||"#94a3b8"}}/>{s}</span>};

function EventFeed({events}:{events:Evt[]}){
  if(!events.length) return null;
  const last = events.slice(-6).reverse();
  const styles:Record<string,{bg:string;border:string;icon:string;text:string}>={
    BANKRUPTCY:{bg:"#fef2f2",border:"#fecaca",icon:"🚨",text:"text-red-700"},
    DEMOTION:{bg:"#fffbeb",border:"#fde68a",icon:"⚠️",text:"text-amber-700"},
    UPGRADE:{bg:"#ecfdf5",border:"#a7f3d0",icon:"🎉",text:"text-emerald-700"},
    TEST_ETH_TOPUP:{bg:"#eef2ff",border:"#c7d2fe",icon:"💰",text:"text-violet-700"},
  };
  return(
    <div className="space-y-1.5 mb-6">
      <h3 className="text-[11px] font-bold text-slate-400 uppercase tracking-wider mb-2">Protocol Events</h3>
      {last.map((e,i)=>{
        const s=styles[e.type]||{bg:"#eff6ff",border:"#bfdbfe",icon:"📋",text:"text-blue-700"};
        return(<div key={i} className="rounded-xl px-4 py-2.5 text-xs" style={{background:s.bg,border:`1px solid ${s.border}`}}>
          <span className="mr-2">{s.icon}</span><span className={`font-bold mr-2 ${s.text}`}>{e.type}</span><span className="text-slate-600">{e.message}</span>
        </div>);
      })}
    </div>);
}

function EconomyTab({eco,ts,events}:{eco:Economy|null;ts:any;events:Evt[]}){
  if(!eco)return<div className="flex items-center justify-center h-64 text-slate-400 text-sm">Waiting for first round…</div>;
  const sD=(ts?.safety||[]).map((v:number,i:number)=>({r:i+1,v}));
  const bD=(ts?.balance||[]).map((v:number,i:number)=>({r:i+1,v}));
  const fD=(ts?.rewards||[]).map((v:number,i:number)=>({r:i+1,rw:v,pn:ts?.penalties?.[i]||0}));
  return(<div className="space-y-6">
    <EventFeed events={events}/>
    <div className="grid grid-cols-2 md:grid-cols-3 lg:grid-cols-6 gap-4">
      <Stat label="Safety S(P)" value={`${(eco.aggregate_safety*100).toFixed(1)}%`} pulse/>
      <Stat label="Active Agents" value={String(eco.active_agents)}/>
      <Stat label="Total Balance" value={`Ξ ${eco.total_balance.toFixed(4)}`} sub="ETH"/>
      <Stat label="Total Earned" value={`Ξ ${eco.total_earned.toFixed(4)}`} sub="ETH"/>
      <Stat label="Completed" value={String(eco.contracts_completed)} sub={`${eco.contracts_failed} failed`}/>
      <Stat label="Chain" value="0G Galileo" sub="Testnet"/>
    </div>
    <div className="grid grid-cols-1 lg:grid-cols-3 gap-5">
      <Card className="p-5"><h3 className="text-[11px] font-bold text-slate-400 uppercase tracking-wider mb-4">Aggregate Safety — Theorem 3</h3>
        <ResponsiveContainer width="100%" height={200}><AreaChart data={sD}>
          <defs><linearGradient id="gS" x1="0" y1="0" x2="0" y2="1"><stop offset="0%" stopColor={VIOLET} stopOpacity={.2}/><stop offset="100%" stopColor={VIOLET} stopOpacity={0}/></linearGradient></defs>
          <CartesianGrid strokeDasharray="3 3" stroke="#f1f5f9"/><XAxis dataKey="r" tick={{fontSize:9,fill:"#94a3b8"}}/><YAxis domain={[0.3,1]} tick={{fontSize:9,fill:"#94a3b8"}}/><Tooltip contentStyle={tt}/>
          <Area type="monotone" dataKey="v" stroke={VIOLET} fill="url(#gS)" strokeWidth={2.5} dot={false}/>
        </AreaChart></ResponsiveContainer></Card>
      <Card className="p-5"><h3 className="text-[11px] font-bold text-slate-400 uppercase tracking-wider mb-4">Circulating ETH</h3>
        <ResponsiveContainer width="100%" height={200}><AreaChart data={bD}>
          <defs><linearGradient id="gB" x1="0" y1="0" x2="0" y2="1"><stop offset="0%" stopColor={BLUE} stopOpacity={.2}/><stop offset="100%" stopColor={BLUE} stopOpacity={0}/></linearGradient></defs>
          <CartesianGrid strokeDasharray="3 3" stroke="#f1f5f9"/><XAxis dataKey="r" tick={{fontSize:9,fill:"#94a3b8"}}/><YAxis tick={{fontSize:9,fill:"#94a3b8"}}/><Tooltip contentStyle={tt}/>
          <Area type="monotone" dataKey="v" stroke={BLUE} fill="url(#gB)" strokeWidth={2.5} dot={false}/>
        </AreaChart></ResponsiveContainer></Card>
      <Card className="p-5"><h3 className="text-[11px] font-bold text-slate-400 uppercase tracking-wider mb-4">Rewards vs Penalties</h3>
        <ResponsiveContainer width="100%" height={200}><AreaChart data={fD}>
          <CartesianGrid strokeDasharray="3 3" stroke="#f1f5f9"/><XAxis dataKey="r" tick={{fontSize:9,fill:"#94a3b8"}}/><YAxis tick={{fontSize:9,fill:"#94a3b8"}}/><Tooltip contentStyle={tt}/>
          <Area type="monotone" dataKey="rw" stroke={GREEN} fill={GREEN} fillOpacity={.1} strokeWidth={2} dot={false} name="Reward"/>
          <Area type="monotone" dataKey="pn" stroke={RED} fill={RED} fillOpacity={.1} strokeWidth={2} dot={false} name="Penalty"/>
        </AreaChart></ResponsiveContainer></Card>
    </div>
  </div>);
}

function AgentsTab({agents}:{agents:Agent[]}){
  const[sort,setSort]=useState<"earned"|"tier"|"balance">("earned");
  if(!agents.length)return<div className="flex items-center justify-center h-64 text-slate-400">No agents yet…</div>;
  const s=[...agents].sort((a,b)=>sort==="tier"?b.current_tier-a.current_tier||b.total_earned-a.total_earned:sort==="balance"?b.balance-a.balance:b.total_earned-a.total_earned);
  const tierCounts:Record<number,number>={};agents.forEach(a=>{tierCounts[a.current_tier]=(tierCounts[a.current_tier]||0)+1});
  const pieData=Object.entries(tierCounts).map(([t,c])=>({name:`T${t}`,value:c,fill:TC[Number(t)]||"#94a3b8"}));
  const robData=agents.filter(a=>a.robustness).map(a=>({name:a.model_name.length>12?a.model_name.slice(0,12)+"…":a.model_name,CC:Math.round((a.robustness?.cc||0)*100),ER:Math.round((a.robustness?.er||0)*100),AS:Math.round((a.robustness?.as_||0)*100)}));
  return(<div className="space-y-6">
    <div className="grid grid-cols-1 lg:grid-cols-2 gap-5">
      <Card className="p-5"><h3 className="text-[11px] font-bold text-slate-400 uppercase tracking-wider mb-4">Tier Distribution</h3>
        <ResponsiveContainer width="100%" height={200}><PieChart><Pie data={pieData} dataKey="value" nameKey="name" cx="50%" cy="50%" innerRadius={50} outerRadius={80} paddingAngle={3} label={({name,value})=>`${name}: ${value}`} labelLine={false} fontSize={11}>
          {pieData.map((e,i)=><Cell key={i} fill={e.fill} stroke="none"/>)}</Pie><Tooltip contentStyle={tt}/></PieChart></ResponsiveContainer></Card>
      <Card className="p-5"><h3 className="text-[11px] font-bold text-slate-400 uppercase tracking-wider mb-4">Robustness Profile</h3>
        <ResponsiveContainer width="100%" height={200}><BarChart data={robData} barGap={1} barSize={8}>
          <CartesianGrid strokeDasharray="3 3" stroke="#f1f5f9"/><XAxis dataKey="name" tick={{fontSize:8,fill:"#94a3b8"}} interval={0} angle={-20} textAnchor="end" height={50}/><YAxis domain={[0,100]} tick={{fontSize:9,fill:"#94a3b8"}}/><Tooltip contentStyle={tt}/>
          <Bar dataKey="CC" fill={VIOLET} radius={[2,2,0,0]}/><Bar dataKey="ER" fill={AMBER} radius={[2,2,0,0]}/><Bar dataKey="AS" fill={BLUE} radius={[2,2,0,0]}/>
        </BarChart></ResponsiveContainer></Card>
    </div>
    <div className="flex items-center gap-2">
      <span className="text-[10px] text-slate-400 uppercase tracking-wider font-semibold">Sort</span>
      {(["earned","tier","balance"] as const).map(x=>(<button key={x} onClick={()=>setSort(x)} className={`text-[11px] px-3 py-1 rounded-full border font-semibold capitalize transition-all ${sort===x?"border-violet-300 text-violet-700 bg-violet-50 shadow-sm":"border-slate-200 text-slate-400 hover:text-slate-600 hover:border-slate-300"}`}>{x}</button>))}
    </div>
    <Card className="overflow-hidden">
      <table className="w-full text-sm">
        <thead><tr className="text-[10px] text-slate-400 uppercase tracking-wider border-b border-slate-100 bg-slate-50/60">
          <th className="px-5 py-3 text-left">Agent</th><th className="px-3 py-3 text-left">Strategy</th><th className="px-3 py-3 text-center">Tier</th>
          <th className="px-3 py-3 text-right">Balance</th><th className="px-3 py-3 text-right">Earned</th><th className="px-3 py-3 text-right">Penalties</th>
          <th className="px-3 py-3 text-center">W/L</th><th className="px-3 py-3 text-left" style={{minWidth:200}}>Robustness</th><th className="px-3 py-3 text-center">⬤</th>
        </tr></thead>
        <tbody>{s.map(a=>(
          <tr key={a.agent_id} className="border-b border-slate-50 hover:bg-violet-50/30 transition-colors">
            <td className="px-5 py-3.5"><div className="font-bold text-slate-800">{a.model_name}</div>{a.ens_name&&<a href={`https://sepolia.app.ens.domains/${a.ens_name}`} target="_blank" rel="noopener noreferrer" className="text-violet-500 font-mono text-[10px] hover:underline">{a.ens_name}</a>}{a.wallet_address&&<div><a href={`https://chainscan-galileo.0g.ai/address/${a.wallet_address}`} target="_blank" rel="noopener noreferrer" className="text-slate-400 font-mono text-[10px] hover:text-violet-500 hover:underline">{a.wallet_address.slice(0,6)}…{a.wallet_address.slice(-4)}</a></div>}</td>
            <td className="px-3 py-3.5 text-slate-500 capitalize text-xs font-medium">{a.strategy}</td>
            <td className="px-3 py-3.5 text-center"><TB t={a.current_tier}/></td>
            <td className="px-3 py-3.5 text-right font-mono text-xs text-slate-700">Ξ {a.balance.toFixed(4)}</td>
            <td className="px-3 py-3.5 text-right font-mono text-xs font-bold text-emerald-600">Ξ {a.total_earned.toFixed(4)}</td>
            <td className="px-3 py-3.5 text-right font-mono text-xs text-red-500">{a.total_penalties.toFixed(4)}</td>
            <td className="px-3 py-3.5 text-center text-xs"><span className="text-emerald-600 font-bold">{a.contracts_completed}</span><span className="text-slate-300 mx-0.5">/</span><span className="text-red-500">{a.contracts_failed}</span></td>
            <td className="px-3 py-3.5">{a.robustness?<div className="space-y-0.5"><RB l="CC" v={a.robustness.cc}/><RB l="ER" v={a.robustness.er}/><RB l="AS" v={a.robustness.as_}/><RB l="IH" v={a.robustness.ih}/></div>:<span className="text-slate-300">—</span>}</td>
            <td className="px-3 py-3.5 text-center"><span className={`inline-block w-2.5 h-2.5 rounded-full ${a.status==="active"?"bg-emerald-500":"bg-slate-300"}`}/></td>
          </tr>))}</tbody>
      </table>
    </Card>
  </div>);
}

function TradesTab({trades}:{trades:Trade[]}){
  const[exp,setExp]=useState<Set<number>>(new Set());
  if(!trades.length)return<div className="flex items-center justify-center h-64 text-slate-400">No trades yet…</div>;
  const sorted=[...trades].reverse();
  const tog=(i:number)=>{setExp(p=>{const n=new Set(p);n.has(i)?n.delete(i):n.add(i);return n})};
  const passed=trades.filter(t=>t.passed).length;
  return(<div className="space-y-4">
    <div className="grid grid-cols-3 gap-4">
      <Stat label="Total Trades" value={String(trades.length)}/>
      <Stat label="Passed" value={String(passed)} sub={`${trades.length?(passed/trades.length*100).toFixed(0):"0"}% pass rate`}/>
      <Stat label="Failed" value={String(trades.length-passed)}/>
    </div>
    <div className="space-y-2">
    {sorted.map((t,i)=>{const o=exp.has(i);const tn=parseInt(t.tier.replace("T",""))||0;return(
      <Card key={i} className={`overflow-hidden transition-all ${o?"ring-2 ring-violet-200":""}`}>
        <button onClick={()=>tog(i)} className="w-full flex items-center gap-3 px-5 py-3.5 text-left hover:bg-violet-50/30 transition-colors">
          <span className="text-[10px] text-slate-400 font-mono w-8">R{t.round+1}</span>
          <span className={`text-[10px] font-extrabold w-10 ${t.passed?"text-emerald-600":"text-red-500"}`}>{t.passed?"PASS":"FAIL"}</span>
          <span className="text-xs font-bold w-44 truncate text-slate-800">{t.agent}</span>
          <TB t={tn}/>
          <span className="text-[10px] text-slate-400 w-20 capitalize font-medium">{t.domain}</span>
          <span className="text-[11px] text-slate-500 flex-1 truncate">{t.task_id}</span>
          <span className="text-xs font-mono font-bold w-28 text-right" style={{color:t.passed?GREEN:RED}}>
            {t.passed?`+Ξ ${t.reward.toFixed(4)}`:`-Ξ ${t.penalty.toFixed(4)}`}
          </span>
          <svg className={`w-4 h-4 text-slate-400 transition-transform ${o?"rotate-180":""}`} fill="none" viewBox="0 0 24 24" stroke="currentColor"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19 9l-7 7-7-7"/></svg>
        </button>
        {o&&(
          <div className="border-t border-slate-100 px-5 py-4 bg-slate-50/40 space-y-4">
            <div className="grid grid-cols-2 md:grid-cols-4 gap-4 text-xs">
              <div><p className="text-[10px] text-slate-400 font-semibold mb-0.5">Task</p><p className="text-slate-700">{t.task_id}</p></div>
              <div><p className="text-[10px] text-slate-400 font-semibold mb-0.5">Domain / Tier</p><p className="text-slate-700 capitalize">{t.domain} · <TB t={tn}/></p></div>
              <div><p className="text-[10px] text-slate-400 font-semibold mb-0.5">Token Cost</p><p className="font-mono text-slate-700">Ξ {t.token_cost.toFixed(6)}</p></div>
              <div><p className="text-[10px] text-slate-400 font-semibold mb-0.5">Latency</p><p className="text-slate-700">{t.latency_ms.toFixed(0)} ms</p></div>
            </div>
            {t.task_prompt&&<div><p className="text-[10px] text-slate-400 font-semibold mb-1.5">Task Definition</p><pre className="text-[11px] text-slate-600 bg-white rounded-xl p-3.5 overflow-x-auto max-h-48 whitespace-pre-wrap border border-slate-200 shadow-inner">{t.task_prompt}</pre></div>}
            {(t.constraints_passed.length>0||t.constraints_failed.length>0)&&(
              <div><p className="text-[10px] text-slate-400 font-semibold mb-1.5">Constraints</p>
                <div className="flex flex-wrap gap-1.5">
                  {t.constraints_passed.map((c,j)=><span key={`p${j}`} className="px-2 py-0.5 rounded-full text-[10px] font-semibold bg-emerald-50 text-emerald-700 border border-emerald-200">✓ {c}</span>)}
                  {t.constraints_failed.map((c,j)=><span key={`f${j}`} className="px-2 py-0.5 rounded-full text-[10px] font-semibold bg-red-50 text-red-600 border border-red-200">✗ {c}</span>)}
                </div></div>)}
            <div><p className="text-[10px] text-slate-400 font-semibold mb-1.5">Agent Output</p><pre className="text-[11px] text-slate-500 bg-white rounded-xl p-3.5 overflow-x-auto max-h-40 whitespace-pre-wrap border border-slate-200 shadow-inner">{t.output_preview}</pre></div>
          </div>)}
      </Card>);})}
    </div>
  </div>);
}

function OnChainTab(){
  const contracts = usePoll<any>(`${API}/api/contracts`, 10000);
  const explorer = contracts?.explorer || "https://chainscan-galileo.0g.ai";
  const registry = contracts?.contracts?.CGAERegistry?.address || "";
  const escrow = contracts?.contracts?.CGAEEscrow?.address || "";
  return(<div className="space-y-5">
    <Card className="p-6"><h3 className="text-sm font-bold text-slate-800 mb-4">0G Chain — Galileo Testnet</h3>
      <div className="grid grid-cols-1 md:grid-cols-2 gap-5 text-sm">
        <div><p className="text-[10px] text-slate-400 font-semibold mb-1">CGAERegistry</p><code className="text-xs text-violet-600 break-all font-semibold">{registry||"Not deployed"}</code></div>
        <div><p className="text-[10px] text-slate-400 font-semibold mb-1">CGAEEscrow</p><code className="text-xs text-violet-600 break-all font-semibold">{escrow||"Not deployed"}</code></div>
        <div><p className="text-[10px] text-slate-400 font-semibold mb-1">Chain ID</p><p className="text-slate-700">16602</p></div>
        <div><p className="text-[10px] text-slate-400 font-semibold mb-1">Architecture</p><p className="text-slate-500 text-xs">Agent wallets · ETH escrow · Weakest-link gate on-chain · Budget ceiling enforcement</p></div>
      </div>
      {registry&&<div className="mt-5"><a href={`${explorer}/address/${registry}`} target="_blank" rel="noopener noreferrer" className="inline-flex items-center gap-2 text-xs px-4 py-2.5 rounded-xl font-bold text-white bg-gradient-to-r from-violet-600 to-blue-600 hover:from-violet-700 hover:to-blue-700 shadow-md shadow-violet-200 transition-all">View on 0G Explorer ↗</a></div>}
    </Card>
    <Card className="p-6"><h3 className="text-sm font-bold text-slate-800 mb-3">Verification Flow</h3>
      <div className="flex items-center gap-2 text-xs text-slate-500 flex-wrap">
        {["audit_live()","→","[CC, ER, AS, IH]","→","0G Storage","→","Merkle root hash","→","CGAERegistry.certify()","→","Tier assigned"].map((step,i)=>(
          <span key={i} className={i%2===0?"text-slate-700 font-semibold bg-slate-100 px-2 py-0.5 rounded-md":"text-violet-400 font-bold"}>{step}</span>))}
      </div>
      <p className="text-xs text-slate-400 mt-3">Anyone can fetch the root hash from the registry, download from 0G Storage, verify the Merkle proof, and confirm scores match.</p>
    </Card>
  </div>);
}

export default function Dashboard(){
  const[tab,setTab]=useState<"economy"|"agents"|"trades"|"onchain">("economy");
  const st=usePoll<{status:string;round:number;total_rounds:number;economy:Economy|null}>(`${API}/api/state`,POLL_MS);
  const ag=usePoll<{agents:Agent[]}>(`${API}/api/agents`,POLL_MS);
  const tr=usePoll<{trades:Trade[]}>(`${API}/api/trades?limit=200`,POLL_MS);
  const ts=usePoll<any>(`${API}/api/timeseries`,POLL_MS);
  const ev=usePoll<{events:Evt[]}>(`${API}/api/events?limit=200`,POLL_MS);
  const status=st?.status||"connecting";
  const round=st?.round||0;
  const totalR=st?.total_rounds||0;
  const tabs=[{id:"economy" as const,l:"📈 Economy"},{id:"agents" as const,l:`🛡️ Agents${ag?.agents?` (${ag.agents.length})`:""}`},{id:"trades" as const,l:`⚡ Trades${tr?.trades?` (${tr.trades.length})`:""}`},{id:"onchain" as const,l:"🔗 On-Chain"}];

  return(<div className="min-h-screen" style={{background:"linear-gradient(135deg, #ede9fe 0%, #e0f2fe 30%, #faf5ff 60%, #fce7f3 100%)"}}>
    <header className="bg-white/70 backdrop-blur-md border-b border-white/50 px-6 py-3.5 sticky top-0 z-50 shadow-sm">
      <div className="max-w-7xl mx-auto flex items-center justify-between">
        <div className="flex items-center gap-5">
          <div className="flex items-center gap-2.5">
            <div className="w-8 h-8 rounded-xl bg-gradient-to-br from-violet-600 to-blue-600 flex items-center justify-center text-white text-sm font-black shadow-lg shadow-violet-200">◆</div>
            <div><h1 className="text-sm font-extrabold text-slate-800">CGAE</h1><p className="text-[10px] text-slate-400 font-medium -mt-0.5">Agent Economy</p></div>
          </div>
          <div className="hidden sm:flex items-center gap-3 text-xs border-l border-slate-200 pl-5">
            <StatusDot s={status}/>
            {round>0&&<span className="text-slate-500">Round <span className="font-bold text-slate-800">{round}</span>{totalR>0?`/${totalR}`:""}</span>}
          </div>
        </div>
        <a href="https://chainscan-galileo.0g.ai" target="_blank" rel="noopener noreferrer" className="text-[11px] px-4 py-2 rounded-xl font-bold text-white bg-gradient-to-r from-violet-600 to-blue-600 hover:from-violet-700 hover:to-blue-700 shadow-md shadow-violet-200 transition-all">0G Explorer ↗</a>
      </div>
    </header>
    <nav className="bg-white/50 backdrop-blur-md border-b border-white/40 px-6 sticky top-[57px] z-40">
      <div className="max-w-7xl mx-auto flex">
        {tabs.map(t=>(<button key={t.id} onClick={()=>setTab(t.id)} className={`px-5 py-3 text-xs font-bold border-b-2 transition-all ${tab===t.id?"border-violet-500 text-violet-700":"border-transparent text-slate-400 hover:text-slate-600"}`}>{t.l}</button>))}
      </div>
    </nav>
    <main className="max-w-7xl mx-auto px-6 py-6">
      {tab==="economy"&&<EconomyTab eco={st?.economy||null} ts={ts} events={ev?.events||[]}/>}
      {tab==="agents"&&<AgentsTab agents={ag?.agents||[]}/>}
      {tab==="trades"&&<TradesTab trades={tr?.trades||[]}/>}
      {tab==="onchain"&&<OnChainTab/>}
    </main>
    <footer className="text-center text-[10px] text-slate-400 font-medium py-5">CGAE · Comprehension-Gated Agent Economy · ETH OpenAgents 2026</footer>
  </div>);
}
