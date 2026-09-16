import { afterEach, describe, expect, it, vi } from 'vitest';
import { ApiClient, tokenStore } from './client';

describe('API client', () => {
  afterEach(() => { vi.unstubAllGlobals(); tokenStore.clear(); });
  it('sends the bearer token and parses JSON', async () => {
    tokenStore.set({ access_token:'access', refresh_token:'refresh', token_type:'bearer', expires_in:900 });
    const fetchMock=vi.fn().mockResolvedValue(new Response(JSON.stringify({status:'ok'}),{status:200})); vi.stubGlobal('fetch',fetchMock);
    await expect(new ApiClient().request<{status:string}>('/health')).resolves.toEqual({status:'ok'});
    expect(fetchMock.mock.calls[0][1].headers.get('Authorization')).toBe('Bearer access');
  });
});
