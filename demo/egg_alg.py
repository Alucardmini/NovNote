# -*- coding:utf-8 -*-
# @version: 1.0
# @author: wuxikun
# @date: '2021/4/1 9:06 PM'

# K个鸡蛋N层楼
def superEggDrop(K: int, N: int):

    memo = dict()

    def dp(K, N):

        if K==1:
            return N
        if N==0:
            return 0

        if (K, N) in memo:
            return memo[(K, N)]

        res = float("INF")

        for i in range(1, N+1):

            res = min(res, max(
                dp(K, N-i),
                dp(K-1, i-1)
            ) + 1)

        memo[(K, N)] = res
        return res

    return dp(K, N)

if __name__ == '__main__':

    print(superEggDrop(2, 100))