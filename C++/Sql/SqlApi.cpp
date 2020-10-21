#include <iostream>
#include <sqlapi.h>
#include <io.h>
#include <fstream>
#include <time.h>  
#include <stdlib.h> 
#include <map>
#include <vector>
#include<list>

using namespace std;

// 贝叶斯估计验证测试
int main() {
	SAConnection con; // connection object
	SACommand cmd;    // create command object

	int i = -20;
	unsigned j = 10;

	cout << i + j << endl;
	return 0;

	ofstream fout("data2.txt");

	Field f;

	f.AbleNull = false;

	try {
		// ODBC 数据源
		con.Connect("regress", "sa", "521374335", SA_ODBC_Client);

		cmd.setConnection(&con);

		//读取
		//cmd.setCommandText("select * from users");//此函数用于设置sql语句
		//cmd.Execute();//执行sql语句
		//con.Commit();
		////while (cmd.FetchNext()) {
		////	fout << cmd.Field("id").asLong() << endl;//输出整型fid
		////	fout << (const char*)cmd.Field("name").asString() << endl;//输出字符型Name
		////}

		//录入
		SACommand insert(&con, "insert into users (name, online, activeState, updateTime) VALUES (:1, :2, :3, :4)");

		//
		string name1[] = { "A","A","A","B","B","C" };

		//1 -> 1/5   2 -> 1/10   3 -> 3/10   4 -> 1/10   5 -> 1/5   6 -> 1/10
		string nameA[] = { "1","1","2","3","3","3","4","5","5","6" };

		//1 -> 1/10   2 -> 1/5   3 -> 3/10   4 -> 1/5   5 -> 1/10   6 -> 1/10
		string nameB[] = { "1","2","2","3","3","3","4","4","5","6" };

		//1 -> 1/10   2 -> 1/5   3 -> 1/10   4 -> 1/10  5 -> 1/2   6 -> 0
		string nameC[] = { "1","2","2","3","4","5","5","5","5","5" };


		//录入1百万条数据
		long int i = 1000000;
		long int num = i;
		char buf[20] = { 0 };
		while (i--) {
			
			//随机姓名
			int num1 = rand() % 6;
			int num2 = rand() % 10;
			
			// 随机 A B C
			string name = name1[num1];

			// 根据 A B C 随机 1 2 3 4 5 6
			if (num1 >= 0 && num1 <= 2) name += nameA[num2];
			else if (num1 > 2 && num1 <= 4) name += nameB[num2];
			else if (num1 > 4 && num1 <= 5) name += nameC[num2];

			//当前时间
			time_t t;  //秒时间  
			tm* local; //本地时间
			t = time(NULL); //获取目前秒时间 
			local = localtime(&t); //转为本地时间
			strftime(buf, 20, "%Y-%m-%d %H:%M:%S", local);

			//数据库录入
			//insert << name.c_str() << 1L << 1L << buf;
			//insert.Execute();

			if (i % 1000 == 0) cout << (float)(num - i) / num << "%" << endl;
		}

		//cout << "影响条数 ： " << insert.RowsAffected() << endl;

	}
	catch (SAException& x) {

		cout << (const char*)x.ErrText() << endl;

		fout << (const char*)x.ErrText() << endl;


		try {
			con.Rollback();
		}
		catch (SAException&) {
		}
	}


	fout.close();
}