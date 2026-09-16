import type { ApiError, Tokens } from '../../types';
const base=(import.meta.env.VITE_API_BASE_URL || 'http://localhost:8000/api/v1').replace(/\/$/,'');
const key='sovereignai.tokens';
export const tokenStore={get:():Tokens|undefined=>{try{const v=localStorage.getItem(key);return v?JSON.parse(v):undefined}catch{return undefined}},set:(t:Tokens)=>localStorage.setItem(key,JSON.stringify(t)),clear:()=>localStorage.removeItem(key)};
export class ApiClient {
 private refreshPromise:Promise<boolean>|null=null;
 async refresh(){ if(this.refreshPromise)return this.refreshPromise; this.refreshPromise=(async()=>{const t=tokenStore.get();if(!t)return false;const r=await fetch(base+'/auth/refresh',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({refresh_token:t.refresh_token})});if(!r.ok){tokenStore.clear();return false}tokenStore.set(await r.json());return true})();try{return await this.refreshPromise}finally{this.refreshPromise=null} }
 async request<T>(path:string, init:RequestInit={}, retry=true):Promise<T>{ const headers=new Headers(init.headers); const token=tokenStore.get();if(token)headers.set('Authorization',`Bearer ${token.access_token}`); if(init.body && !(init.body instanceof FormData) && !headers.has('Content-Type'))headers.set('Content-Type','application/json'); const r=await fetch(base+path,{...init,headers}); if(r.status===401&&retry&&await this.refresh())return this.request<T>(path,init,false);if(!r.ok){let message='Request failed';try{const b=await r.json();message=b.detail||b.message||message}catch{}throw {status:r.status,message,requestId:r.headers.get('X-Request-ID')||undefined} satisfies ApiError} if(r.status===204)return undefined as T;return r.json() as Promise<T> }
 url(path:string){return base+path}
 artifactUrl(filename:string){return `${base}/artifacts/${encodeURIComponent(filename)}`}
}
export const apiClient=new ApiClient();
