import numpy as np

# # Small example to verify interleaving
# num_samples = 2  # 2 sampled points
# num_cp = 5       # 3 control points

# # Create simple test arrays where values are easy to track
# # dcp_x[i, j, :] = [100*i + 10*j + 1, 100*i + 10*j + 2, 100*i + 10*j + 3]  (x derivatives)
# # dcp_y[i, j, :] = [100*i + 10*j + 4, 100*i + 10*j + 5, 100*i + 10*j + 6]  (y derivatives)
# # dcp_z[i, j, :] = [100*i + 10*j + 7, 100*i + 10*j + 8, 100*i + 10*j + 9]  (z derivatives)

# dcp_x = np.zeros((num_samples, num_cp, 3))
# dcp_y = np.zeros((num_samples, num_cp, 3))
# dcp_z = np.zeros((num_samples, num_cp, 3))

# for i in range(num_samples):
#     for j in range(num_cp):
#         base = 100*i + 10*j
#         dcp_x[i, j, :] = [base + 1, base + 2, base + 3]
#         dcp_y[i, j, :] = [base + 4, base + 5, base + 6]
#         dcp_z[i, j, :] = [base + 7, base + 8, base + 9]

# print("dcp_x shape:", dcp_x.shape)
# print("dcp_y shape:", dcp_y.shape)
# print("dcp_z shape:", dcp_z.shape)

# print("\n--- Original arrays ---")
# print("dcp_x (sample 0):\n", dcp_x[:5,:5,:])
# print("dcp_y (sample 0):\n", dcp_y[:5,:5,:])
# print("dcp_z (sample 0):\n", dcp_z[:5,:5,:])

# # Interleave - stack along axis 2 to interleave x,y,z for each control point
# # Shape after stack: (num_samples, num_cp, 3, 3) where axes are (sample, cp, xyz_deriv, 3-vector)
# dCp = np.stack((dcp_x, dcp_y, dcp_z), axis=2).reshape(num_samples, num_cp * 3, 3)

# print("\n--- Interleaved result ---")
# print("dCp shape:", dCp.shape)
# print("\nFor sample 0, columns should be [dcx1 | dcy1 | dcz1 | dcx2 | dcy2 | dcz2 | dcx3 | dcy3 | dcz3]:")
# print("dCp[0] (each row is a 3-vector column):\n", dCp[0])

# print("\n--- Verification ---")
# print("Column 0 (dcx1):", dCp[0, 0, :], "expected:", dcp_x[0, 0, :])
# print("Column 1 (dcy1):", dCp[0, 1, :], "expected:", dcp_y[0, 0, :])
# print("Column 2 (dcz1):", dCp[0, 2, :], "expected:", dcp_z[0, 0, :])
# print("Column 3 (dcx2):", dCp[0, 3, :], "expected:", dcp_x[0, 1, :])
# print("Column 4 (dcy2):", dCp[0, 4, :], "expected:", dcp_y[0, 1, :])
# print("Column 5 (dcz2):", dCp[0, 5, :], "expected:", dcp_z[0, 1, :])

# # Verify all match
# all_correct = True
# for j in range(num_cp):
#     if not np.allclose(dCp[0, 3*j + 0, :], dcp_x[0, j, :]):
#         all_correct = False
#         print(f"Mismatch at dcx{j+1}")
#     if not np.allclose(dCp[0, 3*j + 1, :], dcp_y[0, j, :]):
#         all_correct = False
#         print(f"Mismatch at dcy{j+1}")
#     if not np.allclose(dCp[0, 3*j + 2, :], dcp_z[0, j, :]):
#         all_correct = False
#         print(f"Mismatch at dcz{j+1}")

# print("\n✓ All correct!" if all_correct else "✗ Some mismatches found")

# print("\n--- Original dCp ---")
# print("dCp shape:", dCp.shape)
# print(dCp)
# print("\nSimple transpose")
# print("dCp transposed shape:", dCp.transpose(0,2,1).shape)
# print(dCp.transpose(0,2,1))
# print("\n Reshaped dCp")
# print("dCp reshaped shape:", dCp.transpose(0,2,1).reshape(num_samples*3, -1).shape)
# print(dCp.transpose(0,2,1).reshape(num_samples*3, -1))

# Small example
import numpy as np

u_samples, v_samples = 3, 4
num_cp_u, num_cp_v = 2, 3

Nu = np.array([[1, 2],
               [3, 4],
               [5, 6]])
Nv = np.array([[10, 20, 30],
               [40, 50, 60],
               [70, 80, 90],
               [100, 110, 120]])

su = np.array([[[0.1, 0.2, 0.3],
                [0.4, 0.5, 0.6],
                [0.7, 0.8, 0.9],
                [1.0, 1.1, 1.2]],

               [[1.3, 1.4, 1.5],
                [1.6, 1.7, 1.8],
                [1.9, 2.0, 2.1],
                [2.2, 2.3, 2.4]],

               [[2.5, 2.6, 2.7],
                [2.8, 2.9, 3.0],
                [3.1, 3.2, 3.3],
                [3.4, 3.5, 3.6]]])

result = np.einsum('ua,vb,uvk->uvabk', Nu, Nv, su).reshape(-1, num_cp_u * num_cp_v, 3)

print("Result shape:", result.shape)  # (12, 6, 3)

#Verify: result[u*v_samples + v, a*num_cp_v + b, :] == Nu[u,a] * Nv[v,b] * su[u,v,:]
u, v, a, b = 0, 0, 0, 1
sample_idx = u * v_samples + v
cp_idx = a * num_cp_v + b
print("Match:", np.allclose(result[sample_idx, cp_idx, :], Nu[u,a] * Nv[v,b] * su[u,v,:]))
print("Vectorized:", result[sample_idx, cp_idx, :])

print(f"N[{u},{a}] * N[{v},{b}] =", Nu[u, a], "*", Nv[v, b], "=", Nu[u, a] * Nv[v, b])
print("su[{},{}] =".format(u, v), su[u, v, :])
print("Product =", Nu[u, a] * Nv[v, b] * su[u, v, :])
print("result[{}, {}, :] =".format(sample_idx, cp_idx), result[sample_idx, cp_idx, :])

# Results reshaped 
result_reshaped = result.reshape(-1, num_cp_u * num_cp_v *3)

print("u=0,v=0")
print(Nu[0])
print(Nv[0])
print(su[0,0])
print(result[0])
print(result_reshaped[0])
print("u=0,v=1")
print(Nu[0])
print(Nv[1])
print(su[0,1])
print(result[1])
print(result_reshaped[1])

#print(result.reshape(-1, num_cp_u * num_cp_v * 3)[0])  # First sample (u=0,v=0)
