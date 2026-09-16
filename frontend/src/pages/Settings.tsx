import {useEffect,useState} from "react";
import {api} from "../services/api";
export default function Settings(){
 const [theme,setTheme]=useState(localStorage.getItem("sovereign_theme")||"system"); const [server,setServer]=useState("Checking…");
 useEffect(()=>{api.health().then(h=>setServer(`${h.status} · database ${h.database}`)).catch(()=>setServer("Server unavailable"))},[]);
 const change=(value:string)=>{setTheme(value);localStorage.setItem("sovereign_theme",value);document.documentElement.dataset.theme=value};
 return <section className="page"><header><p className="eyebrow">PREFERENCES</p><h1>Settings</h1><p className="muted">Appearance settings are kept locally. Server secrets are never shown here.</p></header><article className="panel settings"><h2>Appearance</h2><div className="segmented">{['dark','light','system'].map(value=><button key={value} className={theme===value?'selected':''} onClick={()=>change(value)}>{value}</button>)}</div><h2>Server</h2><p>{server}</p><h2>Security</h2><p className="muted">Your current session is stored only for this browser session.</p><h2>About</h2><p className="muted">SovereignAI 1.0 · Local-first, air-gapped architecture</p></article></section>
}
