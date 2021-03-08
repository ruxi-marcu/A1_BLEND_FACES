#ifndef FILESINDIR_H
#define FILESINDIR_H

void get_files_in_directory(
	std::set<std::string> &out, //list of file names within directory
	const std::string &directory //absolute path to the directory
);

#endif