#include <iostream>
#include <chrono>
#include <thread>

#include <petscksp.h>
#include <petsc.h>


// solver
extern "C" void solver(double *x_data, double *A_data, int *A_indices, int *A_indptr, double *b_data, int shape, const char *options) {

    PetscBool init;
    PetscMPIInt RANK, SIZE;
    Mat A;
    Vec b, x;
    KSP ksp;
    PC pc;    

    // Initialize petsc environment
    PetscInitialized(&init);
    if (init == PETSC_FALSE) {
        PetscInitialize(NULL, NULL, NULL, NULL);
    }
    PetscOptionsInsertString(NULL, options);

    MPI_Comm_rank(PETSC_COMM_WORLD, &RANK);
    MPI_Comm_size(PETSC_COMM_WORLD, &SIZE);

    // Set up parallel matrix A
    MatCreate(PETSC_COMM_WORLD, &A);
    MatSetType(A, MATMPIAIJ);
    MatSetSizes(A, PETSC_DECIDE, PETSC_DECIDE, shape, shape);
    // MatMPIAIJSetPreallocation(A, 1000, NULL, 1000, NULL);
    MatSetUp(A);

    // Set up parallel vector b and x0
    VecCreateMPI(PETSC_COMM_WORLD, PETSC_DECIDE, shape, &b);
    VecDuplicate(b, &x);
    VecSetUp(b);
    VecSetUp(x);
    // Provide user-defined interface
    MatSetFromOptions(A);
    VecSetFromOptions(b);
    VecSetFromOptions(x);

    // set values in A , b and x0
    for (PetscInt i = 0; i < shape; i++) {
        // set values of b
        VecSetValue(b, i, b_data[i], ADD_VALUES);
        VecSetValue(x, i, x_data[i], ADD_VALUES);

        for (PetscInt j = A_indptr[i]; j < A_indptr[i+1]; j++) {
            // set values of A
            MatSetValue(A, i, A_indices[j], A_data[j], ADD_VALUES);
        }
    }
    MatAssemblyBegin(A, MAT_FINAL_ASSEMBLY);
    MatAssemblyEnd(A, MAT_FINAL_ASSEMBLY);
    VecAssemblyBegin(b);
    VecAssemblyEnd(b);
    VecAssemblyBegin(x);
    VecAssemblyEnd(x);

    // Set up ksp
    KSPCreate(PETSC_COMM_WORLD, &ksp);
    KSPSetType(ksp, KSPGMRES);
    KSPSetInitialGuessNonzero(ksp, PETSC_TRUE);
    KSPSetTolerances(ksp, PETSC_DEFAULT, PETSC_DEFAULT, PETSC_DEFAULT, PETSC_DEFAULT);
    KSPSetOperators(ksp, A, A);
    KSPSetUp(ksp);
    // Set up pc
    KSPGetPC(ksp, &pc);
    PCSetType(pc, PCNONE);
    // Provide user-defined interface
    KSPSetFromOptions(ksp);

    // Start solving
    // if (RANK == 0) {
    // printf("start solving...\n");
    // printf("initial guess[0] %f\n", x_data[0]);
    // }
    // auto start = std::chrono::high_resolution_clock::now();
    KSPSolve(ksp, b, x);
    // auto end = std::chrono::high_resolution_clock::now();
    // std::chrono::duration<double> duration = end - start;

    // Get solution in root0
    if (RANK == 0) {
    const double *x_ptr;
    VecGetArrayRead(x, &x_ptr);
    for (int i = 0; i <shape; i++) {
        x_data[i] = x_ptr[i];
    }
    VecRestoreArrayRead(x, &x_ptr);
    delete[] x_ptr;
    }

    // // Get IterationNumber
    // PetscInt iterations;
    // KSPGetIterationNumber(ksp, &iterations);

    // Clean
    VecDestroy(&x);
    VecDestroy(&b);
    MatDestroy(&A);
    KSPDestroy(&ksp);

    // if (RANK == 0) {
    // std::cout << "KSPSolve duration: " << duration.count() * 1000 << " ms" << std::endl;
    // std::cout << "KSPSolve iterationNumber: " << iterations << std::endl;
    // printf("\n\n");
    // }
}
