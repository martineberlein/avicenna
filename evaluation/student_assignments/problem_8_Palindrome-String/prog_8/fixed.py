#User function Template for python3
class Solution:
    def isPalindrome(self, S):
        # code here
        count=1
        start=0
        end=len(S)-1
        while(start<end):
        
            if(S[start]==S[end]):
                count=1
                start+=1
                end-=1
            else:
                count=0
                break
            
        return count


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