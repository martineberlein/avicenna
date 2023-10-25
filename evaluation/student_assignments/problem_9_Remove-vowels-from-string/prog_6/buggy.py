#User function Template for python3
class Solution:
    def removeVowels(self, S):
        # code here
        new_s = S.split()
        new_str = []
        for item in new_s:
            chars = [*item]
            result = ""
            for i in chars:
                if not i in ['a', 'e', 'i', 'o', 'u']:
                    result += i
            new_str.append(result)
        return " ".join(new_str)


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