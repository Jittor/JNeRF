#include <string>
//#include <direct.h>
#include "ARAPDeform.h"
//#include "fileSystemUtility.h"
//#include "MatEngine.h"

int main(int argc, char *argv[])
{
	if (argc == 5)
	{
		std::string inputObj = argv[1];
		std::string handleFile = argv[2];
		std::string outputFolder = argv[3];
		bool hardConstrain = atoi(argv[4]);
		TetrahedralMesh meshOri;
		myReadFile(inputObj.c_str(), meshOri);
		std::string outputName = "test_output.ovm";
		myWriteFile(outputName, meshOri);
		ARAPDeform *arapDeform = new ARAPDeform(meshOri, hardConstrain);
		arapDeform->yyj_ARAPDeform(handleFile, outputFolder);
	}
	else
	{
		std::cout << "exe inputObj handleFile outputFolder hardConstrain" << std::endl;
	}

	return 0;
}