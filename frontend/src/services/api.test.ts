import {describe,expect,it} from "vitest";
import {api,session} from "./api";
describe("API client",()=>{
 it("keeps tokens in session storage and produces backend download URLs",()=>{
  session.set("test-token"); expect(session.get()).toBe("test-token");
  expect(api.downloadUrl("abc","file")).toContain("/api/v1/files/abc/download");
  session.clear(); expect(session.get()).toBeNull();
 });
});
