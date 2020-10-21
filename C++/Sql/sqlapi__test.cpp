#include <iostream>
#include <sqlapi.h>
#include <io.h>
#include <fstream>
#include <time.h>  
#include <stdlib.h> 

using namespace std;
int main() {
	SAConnection con; // connection object
	SACommand cmd;    // create command object

	ofstream fout("data2.txt");

	try {
		// ODBC 数据源
		con.Connect("regress", "sa", "521374335", SA_ODBC_Client);

		cmd.setConnection(&con);

		//读取
		cmd.setCommandText("select * from users");//此函数用于设置sql语句
		cmd.Execute();//执行sql语句
		con.Commit();
		while (cmd.FetchNext()) {
			fout << cmd.Field("id").asLong() << endl;//输出整型fid
			fout << (const char*)cmd.Field("name").asString() << endl;//输出字符型Name
		}

		//录入
		SACommand insert(&con, "insert into users (name, online, activeState, updateTime) VALUES (:1, :2, :3, :4)");


		string name1[] = { "A","B","C","D","E","F","G","H","I" };
		string name2[] = { "1","2","3","4","5","6","7","8","9","10" };

		//录入1百万条数据
		long int i = 1000000;
		char buf[20] = { 0 };
		while (i--) {
			
			//随机姓名
			int num1 = rand() % 9;
			int num2 = rand() % 10;
			string name = name1[num1] + name2[num2];

			//当前时间
			time_t t;  //秒时间  
			tm* local; //本地时间
			t = time(NULL); //获取目前秒时间 
			local = localtime(&t); //转为本地时间
			strftime(buf, 20, "%Y-%m-%d %H:%M:%S", local);

			//数据库录入
			insert << name.c_str() << 1L << 1L << buf;
			insert.Execute();

			if (i % 1000 == 0) cout << "i ： " << i << endl;
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