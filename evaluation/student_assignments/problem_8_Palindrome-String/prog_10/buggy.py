#User function Template for python3
class Solution:
    def isPalindrome(self, S):
        # code here
        n = len(S)
        
        start = 0
        end = n-1
        
        for i in range(n//2):
            
            start_char = S[start]
            end_char = S[end]
            
            if start_char == end_char:
                start += 1
                end -= 1
            else:
                return 0
                
            return 1


#{ 
 # Driver Code Starts
#Initial Template for Python 3

if __name__ == '__main__':
    T=int(input())
    for i in range(T):
        S = input()
        ob = Solution()
        answer = ob.isPalindrome(S)
        print(answer)

# } Driver Code Ends