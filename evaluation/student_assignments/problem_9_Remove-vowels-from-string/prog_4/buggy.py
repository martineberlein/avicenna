#User function Template for python3
class Solution:
    def removeVowels(self, S):
        # code here
        new = " "
        for i in S:
            if not i in "aeiouAEIOU":
                new += i
        return new


#{ 
 # Driver Code Starts
#Initial Template for Python 3

if __name__ == '__main__':
	T=int(input())
	for i in range(T):
		s = input()
		
		ob = Solution()	
		answer = ob.removeVowels(s)
		
		print(answer)


# } Driver Code Ends