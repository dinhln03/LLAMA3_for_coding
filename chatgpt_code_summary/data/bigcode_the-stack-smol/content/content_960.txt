class Solution:
    def backspaceCompare(self, S: str, T: str) -> bool:
        s_bcount, t_bcount = 0, 0
        s_idx, t_idx = len(S) - 1, len(T) - 1
        
        while s_idx >= 0 or t_idx >= 0:
            while s_idx >= 0:
                if S[s_idx] == '#':
                    s_bcount += 1
                    s_idx -= 1
                    continue
                
                if s_bcount > 0:
                    s_idx -= 1
                    s_bcount -= 1
                else:
                    break
                    
            while t_idx >= 0:
                if T[t_idx] == '#':
                    t_bcount += 1
                    t_idx -= 1
                    continue
                
                if t_bcount > 0:
                    t_idx -= 1
                    t_bcount -= 1
                else:
                    break
        
            if s_idx >= 0 and t_idx >= 0 and S[s_idx] != T[t_idx]:
                return False
            elif (s_idx >= 0 and t_idx < 0) or (s_idx < 0 and t_idx >= 0):
                return False
            
            s_idx -= 1
            t_idx -= 1
        
        return True
