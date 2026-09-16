export type ApiError={code:string;message:string;request_id?:string};
const base=import.meta.env.VITE_API_BASE_URL || "http://localhost:8000";
let token:string|null=sessionStorage.getItem("sovereign_token");
export const session={get:()=>token,set:(value:string)=>{token=value;sessionStorage.setItem("sovereign_token",value)},clear:()=>{token=null;sessionStorage.removeItem("sovereign_token")}};
async function request<T>(path:string,init:RequestInit={}):Promise<T>{
 const headers=new Headers(init.headers); if(token) headers.set("Authorization",`Bearer ${token}`); if(init.body && !(init.body instanceof FormData)) headers.set("Content-Type","application/json");
 let response:Response; try{response=await fetch(`${base}${path}`,{...init,headers})}catch{throw {code:"SERVER_UNAVAILABLE",message:"Unable to connect to the SovereignAI server."} satisfies ApiError}
 const json=await response.json().catch(()=>null); if(!response.ok || !json?.success) throw (json?.error || {code:"REQUEST_ERROR",message:"The request could not be completed."}); return json.data as T;
}
export const api={
 health:()=>request<{status:string;database:string;version:string}>("/api/v1/health"),
 register:(body:{email:string;password:string;name:string})=>request<{id:string}>("/api/v1/auth/register",{method:"POST",body:JSON.stringify(body)}),
 login:(body:{email:string;password:string})=>request<{access_token:string;expires_in:number}>("/api/v1/auth/login",{method:"POST",body:JSON.stringify(body)}),
 me:()=>request<{id:string;email:string;name:string;role:string}>("/api/v1/auth/me"),
 chats:()=>request<Array<{id:string;title:string;created_at:string}>>("/api/v1/chats"),
 chat:(id:string)=>request<{id:string;title:string;messages:Array<{id:string;role:string;content:string;model_used?:string;created_at:string}>}>(`/api/v1/chats/${id}`),
 createChat:(title:string)=>request<{id:string;title:string}>("/api/v1/chats",{method:"POST",body:JSON.stringify({title})}),
 message:(id:string,content:string)=>request<{message:{id:string;role:string;content:string;model_used?:string;citations?:Array<Record<string,unknown>>;execution_details?:Record<string,unknown>;artifact?:{filename:string;type:string;size:number}|null};task_id:string}>(`/api/v1/chats/${id}/messages`,{method:"POST",body:JSON.stringify({content})}),
 upload:(file:File)=>{const form=new FormData();form.append("file",file);return request<{id:string;filename:string;size:number;mime_type:string;status:string}>("/api/v1/files/upload",{method:"POST",body:form})},
 document:(id:string)=>request<{id:string;filename:string;mime_type:string;size:number;status:string}>(`/api/v1/documents/${id}`),
 documentContent:(id:string)=>request<{content?:string;message?:string}>(`/api/v1/documents/${id}/content`),
 task:(input:{task_type:string;input:Record<string,unknown>})=>request<{id:string;status:string;output?:{result?:string;model_used?:string};execution_time_ms?:number}>("/api/v1/tasks",{method:"POST",body:JSON.stringify(input)}),
 result:(id:string)=>request<{id:string;status:string;output?:{result?:string;model_used?:string};execution_time_ms?:number;has_download:boolean}>(`/api/v1/results/${id}`),
 downloadUrl:(id:string,kind:"file"|"result")=>`${base}/api/v1/${kind === "file" ? "files" : "results"}/${id}/download`
};
