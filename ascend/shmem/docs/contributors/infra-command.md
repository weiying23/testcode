****

## Ascend社区评论命令一览

Ascend社区中所有项目均由Bot维护，这意味着开发人员可以在每个Pull Request或者Issue下面通过评论触发Bot命令。

<table class="command">
    <thead>
        <tr>
            <th>命令</th>
            <th width="25%">示例</th>
            <th>描述</th>
            <th>面向对象</th>
        </tr>
    </thead>
    <tbody>
        <tr>
            <td>
                /check-cla
            </td>
            <td style="white-space:nowrap;">
                /check-cla
            </td>
            <td>
                强制重新检查Pull Request的CLA状态。
                如果Pull Request的提交者已经签署了CLA协议，则<strong>ascend-cla/yes</strong>标签将会被添加到Pull Request中；如果没有，则标签<strong>ascend-cla/no</strong>将被添加到Pull Request中。
            </td>
            <td>
                所有开发者
            </td>
        </tr>
        <tr>
            <td>
                /cla cancel
            </td>
            <td style="white-space:nowrap;">
                /cla cancel
            </td>
            <td>
                强制删除<strong>ascend-cla/yes</strong>标签。
            </td>
            <td>
               Repo_Admins，仓库维护者，Pull Request提交者
            </td>
        </tr>
        <tr>
           <td>
                compile
           </td>
           <td style="white-space:nowrap;">
                compile
           </td>
           <td>
                触发编译流水线。
                编译通过后，该Pull Request会被打上<strong>ci-pipeline-passed</strong>的标签。若编译失败，该Pull Request会被打上<strong>ci-pipeline-failed</strong>的标签。
           </td>
           <td>
              所有开发者
           </td>
        </tr>
        <tr>
            <td>
                /lgtm [cancel]
            </td>
            <td style="white-space:nowrap;">
                /lgtm
                <br/>
                /lgtm cancel
            </td>
            <td>
                添加或移除用于代表代码已经评审过的标签 <strong>lgtm</strong>。      
            </td>
            <td>
              Repo_Admins，仓库维护者，或者仓库Committers
            </td>
        </tr>
        <tr>
            <td>
                /approve [cancel]
            </td>
            <td style="white-space:nowrap;">
                /approve
                <br/>
                /approve cancel
            </td>
            <td>
                添加或移除用于代表同意合并的标签<strong>approved</strong>。
            </td>
            <td>
              Repo_Admins，仓库维护者，或者Committers
            </td>
        </tr>
        <tr>
            <td>
                /check-pr
            </td>
            <td style="white-space:nowrap;">
                /check-pr
            </td>
            <td>
                检查Pull Request中的标签是否满足条件，如果满足条件，则合并Pull Request。
            </td>
            <td>
                任何人都可以在Pull Request上触发此命令。
            </td>
        </tr>
        <tr>
            <td>
                /merge
            </td>
            <td style="white-space:nowrap;">
                /merge
            </td>
            <td>
                添加用于代表branch_keeper同意合并的标签 <strong>keeper_approved</strong>。
            </td>
            <td>
              仓库对应分支的branch_keeper
            </td>
        <tr>
            <td>
                /kind **
            </td>
            <td style="white-space:nowrap;">
                /kind bug, 
                </br>**可接受大小写字母、数字、中划线、下划线，
                </br>对于下面的**规则通用
            </td>
            <td>
                添加标签 <strong>kind/bug</strong>。
            </td>
            <td>
                仓库管理员可以直接添加；其他人可以使用评论添加标签，如kind/AI，前提是仓库中必须存在此标签，否则添加不上
            </td>
        </tr>
        <tr>
            <td>
                /remove-kind **
            </td>
            <td style="white-space:nowrap;">
                /remove-kind bug
            </td>
            <td>
                移除标签 <strong>kind/bug</strong>。
            </td>
            <td>
                所有人
            </td>
        </tr>
        <tr>
            <td>
                /priority **
            </td>
            <td style="white-space:nowrap;">
                /priority high
            </td>
            <td>
                添加标签 <strong>priority/high</strong>。
            </td>
            <td>
                仓库管理员可以直接添加；其他人可以使用评论添加标签，如kind/AI，前提是仓库中必须存在此标签，否则添加不上
            </td>
        </tr>
        <tr>
            <td>
                /remove-priority **
            </td>
            <td style="white-space:nowrap;">
                /remove-priority high
            </td>
            <td>
                移除标签 <strong>priority/high</strong>。
            </td>
            <td>
                所有人
            </td>
        </tr>
        <tr>
            <td>
                /sig **
            </td>
            <td style="white-space:nowrap;">
                /sig AI
            </td>
            <td>
                添加标签 <strong>sig/AI</strong>。
            </td>
            <td>
                仓库管理员可以直接添加；其他人可以使用评论添加标签，如kind/AI，前提是仓库中必须存在此标签，否则添加不上
            </td>
        </tr>
        <tr>
            <td>
                /remove-sig **
            </td>
            <td style="white-space:nowrap;">
                /remove-sig AI
            </td>
            <td>
                移除标签 <strong>sig/AI</strong>。
            </td>
            <td>
                所有人
            </td>
        </tr>
        <tr>
            <td>
                /assign [[@]...]
            </td>
            <td style="white-space:nowrap;">
                /assign 
                </br>/assign @ascend-robot
            </td>
            <td>
                为Issue指派一位负责人。
            </td>
            <td>
                所有人
            </td>
        </tr>
        <tr>
            <td>
                /unassign [[@]...]
            </td>
            <td style="white-space:nowrap;">
                /unassign 
                </br>/unassign @ascend-robot
            </td>
            <td>
                取消Issue指派的负责人。
            </td>
            <td>
                所有人
            </td>
        </tr>
    </tbody>
</table>