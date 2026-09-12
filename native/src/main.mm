#include "lyra/cli.hpp"
#include "lyra/runtime.hpp"
#include "lyra/storage.hpp"
#import <Foundation/Foundation.h>
#include <iostream>

int main(int argc, char **argv) {
  @autoreleasepool {
    try {
      return lyra::cli_main(argc, argv);
    } catch (const lyra::Error &error) {
      std::cerr << error.type << ": " << error.what() << '\n';
      return error.type == "InterruptedError" ? 130 : 1;
    } catch (const std::exception &error) {
      std::cerr << lyra::exception_type(error) << ": " << error.what() << '\n';
      return 1;
    }
  }
}
